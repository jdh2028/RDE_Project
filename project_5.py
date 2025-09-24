import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import cantera as ct
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

# ==============================================================================
# 0) 노즐 형상 (x: 축방향, y=R: 반경) — throat 기준 x=0
# ==============================================================================
class NozzleContour:
    def __init__(self, contour_data_cm):
        data = np.array(contour_data_cm, dtype=float) / 100.0  # cm -> m
        x, y = data[:, 0], data[:, 1]
        s = np.argsort(x)
        self.x_data, self.y_data = x[s], y[s]
        # throat
        self.throat_idx = np.argmin(self.y_data)
        self.x_throat   = self.x_data[self.throat_idx]
        self.R_t        = self.y_data[self.throat_idx]
        self.D_t        = 2.0 * self.R_t
        self.A_t        = np.pi * self.R_t**2
        # x=0 shift
        self.x_data_shifted = self.x_data - self.x_throat
        # PCHIP + 도함수
        self.spline = PchipInterpolator(self.x_data_shifted, self.y_data, extrapolate=False)
        self.d1     = self.spline.derivative(1)
        self.d2     = self.spline.derivative(2)
        # throat 곡률반경
        self.r_c_throat = self._curvature_radius(0.0)

    def _curvature_radius(self, x):
        dy  = float(self.d1(x))
        d2y = float(self.d2(x))
        denom = (1.0 + dy*dy)**1.5
        kappa = abs(d2y)/denom if denom > 0 else 0.0
        if kappa < 1e-12:
            return 1e9
        return 1.0 / kappa

    def get_radius(self, x):
        R = self.spline(x)
        R = np.where(np.isnan(R), self.R_t, R)
        return np.maximum(R, self.R_t)

    def get_profile(self, x):
        R  = self.get_radius(x)
        A  = np.pi * R**2
        D  = 2.0 * R
        AR = A / self.A_t
        return D, AR

# ==============================================================================
# 1) 리브두께 → 핀효율
# ==============================================================================
class RibThicknessProfile:
    def __init__(self, table_cm_mm):
        arr = np.array(table_cm_mm, dtype=float)
        x_cm, t_mm = arr[:, 0], arr[:, 1]
        s = np.argsort(x_cm)
        x_sorted, t_sorted = x_cm[s], t_mm[s]
        ux, idx = np.unique(x_sorted, return_index=True)
        t_unique = t_sorted[idx]
        self.x_m = ux / 100.0
        self.t_m = t_unique / 1000.0
        self._pchip = PchipInterpolator(self.x_m, self.t_m, extrapolate=False)

    def t_rib(self, x_m):
        return np.maximum(self._pchip(x_m), 1e-4)  # 0.1 mm 하한

def fin_efficiency_eta(h_c, k_wall, t_rib, L_fin, tip_correction=True):
    # 평판핀(양면 대류): m = sqrt(2 h/(k t)), η = tanh(m L_e)/(m L_e)
    t = max(float(t_rib), 1e-6)
    L = max(float(L_fin), 1e-6)
    m  = np.sqrt(2.0 * h_c / (k_wall * t))
    Le = L + 0.5 * t if tip_correction else L
    ml = m * Le
    if ml < 1e-6:
        return 1.0
    return float(np.tanh(ml) / ml)

# ==============================================================================
# 2) Cantera 유틸 (동결 조성, 챔버에서 고정) + c* (choked ideal)
# ==============================================================================
_GAS = ct.Solution('gri30_highT.yaml')
Y_FROZEN = None
GAMMA_CHAMBER = None
R_CHAMBER = None

def set_frozen_composition(of, T0, P0):
    """챔버에서 HP 평형으로 조성 고정 → 전 구간 frozen 물성 사용"""
    global _GAS, Y_FROZEN, GAMMA_CHAMBER, R_CHAMBER
    phi = 4.0 / of  # CH4/O2
    _GAS.set_equivalence_ratio(phi, fuel='CH4', oxidizer='O2')
    _GAS.TP = T0, P0
    _GAS.equilibrate('HP')  # 챔버에서 평형 → 조성 고정
    Y_FROZEN = _GAS.Y.copy()
    cp0 = _GAS.cp_mass; cv0 = _GAS.cv_mass
    GAMMA_CHAMBER = cp0 / cv0
    R_CHAMBER = cp0 - cv0
    return Y_FROZEN, GAMMA_CHAMBER, R_CHAMBER

def gas_props_frozen(T, P):
    """동결 조성(Y_FROZEN)으로 γ, μ, k, cp, Pr 계산"""
    _GAS.TPY = T, P, Y_FROZEN
    cp = _GAS.cp_mass; cv = _GAS.cv_mass
    gamma = cp / cv
    mu = _GAS.viscosity
    k  = _GAS.thermal_conductivity
    Pr = mu * cp / k
    return gamma, mu, cp, k, Pr

def state_with_frozen(T0, P0, AR, flow_type, iters=12):
    """면적비→M 반복(γ는 동결 조성으로 T,P에 의존)"""
    gamma = 1.20
    for _ in range(iters):
        M  = mach_from_area_ratio(AR, gamma, flow_type)
        Ts = T0 / (1.0 + 0.5 * (gamma - 1.0) * M*M)
        Ps = P0 * (Ts / T0)**(gamma / (gamma - 1.0))
        gamma_new, mu, cp, k, Pr = gas_props_frozen(Ts, Ps)
        if abs(gamma_new - gamma) < 1e-4:
            gamma = gamma_new
            break
        gamma = 0.5 * (gamma + gamma_new)
    return M, Ts, Ps, gamma, mu, cp, k, Pr

def cstar_from_chamber(T0, P0):
    """c* = sqrt(R*T0/gamma) * ((γ+1)/2)^{(γ+1)/(2(γ-1))}"""
    gamma = GAMMA_CHAMBER
    R = R_CHAMBER
    return np.sqrt(R*T0/gamma) * ((gamma+1.0)/2.0)**((gamma+1.0)/(2.0*(gamma-1.0)))

# ==============================================================================
# 3) 유동/열전달
# ==============================================================================
def mach_from_area_ratio(area_ratio, gamma=1.4, flow_type='subsonic'):
    f = lambda M: (1.0/M**2) * ((2.0/(gamma+1.0)) * (1.0 + 0.5*(gamma-1.0)*M*M))**((gamma+1.0)/(gamma-1.0)) - area_ratio**2
    if area_ratio < 1.0:
        raise ValueError("A/A*는 1 이상이어야 합니다.")
    if abs(area_ratio - 1.0) < 1e-14:
        return 1.0
    eps = 1e-12
    if flow_type == 'subsonic':
        return brentq(f, eps, 1.0 - eps)
    a, b = 1.0 + eps, 2.0
    while f(b) <= 0.0:
        b *= 2.0
        if b > 1e6:
            raise RuntimeError("상한 탐색 실패")
    return brentq(f, a, b)

# --- Bartz h_g using Eq.(8): centerline(ref) properties, ω=0 ---
def bartz_hg_eq8(mdot_g, A_local, D_t, r_c_throat, mu_ref, cp_ref, k_ref):

    return (
        0.026
        * (mu_ref ** (-0.4))
        * (cp_ref ** 0.4)
        * (k_ref ** 0.6)
        * (mdot_g ** 0.8) / (A_local ** 0.9)
        * ((np.pi * D_t) / (4.0 * r_c_throat)) ** 0.1
    )

def _friction_factor_colebrook(Re, D_h, eps=6e-6):
    if Re < 2300.0:
        return 64.0 / Re
    g = lambda f: 1.0/np.sqrt(f) + 2.0*np.log10(eps/D_h/3.7 + 2.51/(Re*np.sqrt(f)))
    return brentq(g, 1e-6, 0.2)

def calculate_pressure_drop(m_dot_c, T_c, P_c, A_c, d_h, dx, fluid):
    rho = CP.PropsSI('D', 'T', T_c, 'P', P_c, fluid)
    mu  = CP.PropsSI('V', 'T', T_c, 'P', P_c, fluid)
    v   = m_dot_c / (A_c * rho)
    Re  = rho * v * d_h / mu
    f   = _friction_factor_colebrook(Re, d_h, eps=6e-6)
    return float(f * (dx/d_h) * 0.5 * rho * v * v)

def calculate_h_c(m_dot_c, A_c, d_h, T_c, T_wc, P_c, fluid='Methane', eta_fin=1.0):
    # Dittus–Boelter 유지 (Gnielinski 미적용)
    if not (91 < T_c < 600 and 1e5 < P_c < 200e5):
        return 3.0e4 * eta_fin
    rho = CP.PropsSI('D', 'T', T_c, 'P', P_c, fluid)
    cp  = CP.PropsSI('C', 'T', T_c, 'P', P_c, fluid)
    mu  = CP.PropsSI('V', 'T', T_c, 'P', P_c, fluid)
    k   = CP.PropsSI('L', 'T', T_c, 'P', P_c, fluid)
    Pr  = cp * mu / k
    v   = m_dot_c / (A_c * rho)
    Re  = rho * v * d_h / mu
    if Re < 4000:
        return 1.0e3 * eta_fin
    Nu  = 0.023 * (Re**0.8) * (Pr**0.4)
    h   = Nu * k / d_h
    return h * eta_fin

# ==============================================================================
# 4) 입력 (논문 Table 1,2 기반)
# ==============================================================================
nozzle_contour_data = np.array([
    [-35,10.72], [-33.6,10.76], [-31.8,10.65], [-30.2,10.65], [-28.3,10.65],
    [-26.5,10.65], [-24.5,10.64], [-21.6,10.64], [-20.0,10.53], [-17.8,10.32],
    [-15.3, 9.81], [-13.4, 9.40], [-11.3, 8.78],  [-9.0, 7.96],  [-7.4, 7.45],
     [-4.7, 6.53],  [-2.6, 5.91],  [-1.1, 5.71],   [0.05, 5.81],  [1.0, 6.42],
     [2.0, 7.33],   [2.9, 8.05],   [4.1, 8.96],   [5.5, 9.88],   [6.9, 11.0],
     [8.5,12.12],  [10.2,13.24], [11.8,14.15], [13.1,14.96], [14.5,15.88],
    [16.1,16.80], [18.1,17.91], [20.1,19.03], [21.5,19.74], [22.9,20.56],
    [23.8,20.96], [25,21.57]
])

rib_contour_data = np.array([
    [-35,3.4], [-31.0,3.4], [-27.1,3.4], [-23.0,3.4], [-19.2,3.3], [-15.2,3.1],
    [-11.2,2.6], [-7.2,2.1], [-3.2,1.5], [0.5,1.5], [4.5,2.8],
    [8.5,4.0], [12.5,5.1], [16.5,5.1], [20.4,7.0], [25,7.8]
])

heat_flux_data = np.array([
    [-35,13.1], [-28.2,12.8], [-21.5,12.8], [-14.8,14.5], [-8.1,21.3], [-0.9,35.1],
     [4.8,14.2], [11.2,6], [17.7,3.0], [25,1.4]
])

T_wg_data = np.array([
    [-35, 572.0], [-28.3, 537.3], [-21.7, 498.9], [-15.0, 484.0], [-8.2, 532.7], [-1.4, 675.2],
    [4.4, 396.4], [10.8, 261.2], [17.5, 195.9], [25, 154.2]
])

T_c_data = np.array([
    [-35, 312.0], [-27.7, 286.1], [-21.2, 261.7], [-14.7, 239.9], [-8.2, 218.1], [-1.7, 191],
    [4.8, 159.8], [11.3, 136.7], [18.1, 116.1], [25, 95.7]
])

P_c_data = np.array([
    [-35, 122], [-28.3, 123.0], [-21.5, 123.9], [-14.9, 124.6], [-8.5, 125.2], [-1.5, 125.7],
    [4.8, 126.1], [11.2, 126.4], [17.9, 126.7], [25, 127.0]
])

# 챔버 조건 (Table 1)
T_0 = 3603.2       # K
P_0 = 58.6e5       # Pa
O_F = 3.5          # 질량비

# 동결 조성 초기화 + c*
set_frozen_composition(O_F, T_0, P_0)
C_star = cstar_from_chamber(T_0, P_0)

# 냉각 채널/재료 (Table 2)
wall_thickness = 0.7e-3
channel_height = 8.63e-3
channel_width  = 1.08e-3
num_channels   = 150
k_wall         = 300.0      # 구리 라이너 가정
COOLANT_NAME   = 'Methane'
m_dot_coolant  = 6.858          # kg/s
T_c_inlet      = 120.0      # K
P_c_inlet      = 127e5      # Pa
cooling_jacket_end_ar = 15.0

# 파생
nozzle      = NozzleContour(nozzle_contour_data)
rib_profile = RibThicknessProfile(rib_contour_data)

D_t        = nozzle.D_t
A_c_total  = channel_width * channel_height * num_channels
d_h        = 2.0 * (channel_width * channel_height) / (channel_width + channel_height)
r_c_throat = nozzle.r_c_throat

# 가스 질량유량 (Eq.(8)에 필요): \dot{m} = P0 * A* / c*
m_dot_gas = P_0 * nozzle.A_t / C_star

# 해석 세팅
n_segments  = 200
tol         = 1e-5
max_iter    = 1000
alpha_relax = 0.5

# 해석 구간: 수축부 끝 ~ 팽창부 AR=15
x_start = nozzle.x_data_shifted[0]
x_end   = 0.25
for xv in np.linspace(0, 0.5, 500):
    _, ar = nozzle.get_profile(xv)
    if ar >= cooling_jacket_end_ar:
        x_end = xv
        break
x_pos = np.linspace(x_start, x_end, n_segments)
dx    = x_pos[1] - x_pos[0]

# 결과 배열
D_profile   = np.zeros(n_segments)
AR_profile  = np.zeros(n_segments)
T_c_profile = np.zeros(n_segments)
P_c_profile = np.zeros(n_segments)
T_wg_profile= np.zeros(n_segments)
T_wc_profile= np.zeros(n_segments)
q_profile   = np.zeros(n_segments)
h_g_profile = np.zeros(n_segments)

# 경계조건(출구에서 입구로 적분)
T_c_profile[-1] = T_c_inlet
P_c_profile[-1] = P_c_inlet
T_wg_guess_next = 600.0

# ==============================================================================
# 5) 메인 루프 (Eq.(8) 사용)
# ==============================================================================
print("Start 1D regenerative cooling (Bartz Eq.(8), frozen props, rib area) ...")

for i in range(n_segments - 1, -1, -1):
    x = x_pos[i]
    D_local, AR_local = nozzle.get_profile(x)
    D_profile[i], AR_profile[i] = D_local, AR_local
    A_local = np.pi * (0.5 * D_local)**2

    T_c_local, P_c_local = T_c_profile[i], P_c_profile[i]

    flow_type = 'subsonic' if x < 0.0 else 'supersonic'
    M, T_s, P_s, gamma_s, mu_s, cp_s, k_s, Pr_s = state_with_frozen(T_0, P_0, AR_local, flow_type)

    T_wg_iter = float(T_wg_guess_next)
    q = 0.0
    Twc = float(T_c_local)
    Twg_new = float(T_wg_iter)

    for _ in range(max_iter):
        # (a) Eq.(8)으로 h_g 계산 (ref=중심축 Ts, Ps)
        h_g = bartz_hg_eq8(
            mdot_g=m_dot_gas, A_local=A_local, D_t=D_t, r_c_throat=r_c_throat,
            mu_ref=mu_s, cp_ref=cp_s, k_ref=k_s
        )
        # (b) T_aw (복원온도, r = Pr^{1/3})
        r_rec = Pr_s ** (1.0 / 3.0)
        T_aw = T_0 * (1.0 + r_rec * 0.5 * (gamma_s - 1.0) * M*M) / (1.0 + 0.5 * (gamma_s - 1.0) * M** 2)

        # (c) 냉각측: Dittus–Boelter
        h_c_raw = calculate_h_c(m_dot_coolant, A_c_total, d_h, T_c_local, T_wg_iter, P_c_local, COOLANT_NAME, eta_fin=1.0)

        # (d) 핀효율 + 면적증대(피치 보정)
        t_rib_loc = float(rib_profile.t_rib(x))
        eta_fin   = fin_efficiency_eta(h_c_raw, k_wall, t_rib_loc, channel_height, tip_correction=True)
        R_gas     = float(nozzle.get_radius(x))
        pitch = 2.0 * np.pi * (R_gas + wall_thickness + 0.5*channel_height)/ num_channels
        area_ratio = 1.0 + eta_fin * (2.0 * channel_height) / pitch
        h_c_eff = h_c_raw * area_ratio

        # (e) 열저항망
        Rg = 1.0 / max(h_g, 1e-12)
        Rw = wall_thickness / k_wall
        Rc = 1.0 / h_c_eff

        q       = max((T_aw - T_c_local) / (Rg + Rw + Rc), 0.0)  # [W/m^2]
        Twc     = T_c_local + q * Rc
        Twg_new = T_aw - q * Rg

        if not np.isfinite(Twg_new):
            break
        if abs(Twg_new - T_wg_iter) < tol:
            break
        T_wg_iter = (1.0 - alpha_relax) * T_wg_iter + alpha_relax * Twg_new

    T_wg_profile[i] = Twg_new
    T_wc_profile[i] = Twc
    q_profile[i]    = q
    h_g_profile[i]  = h_g
    T_wg_guess_next = Twg_new

    # 냉각수 다음 구간 상태
    if i > 0:
        A_ht  = np.pi * D_local * dx
        Q_seg = q * A_ht
        Cp_c  = CP.PropsSI('C', 'T', T_c_local, 'P', P_c_local, COOLANT_NAME)
        T_c_profile[i-1] = T_c_local + Q_seg / (m_dot_coolant * Cp_c)
        dP = calculate_pressure_drop(m_dot_coolant, T_c_local, P_c_local, A_c_total, d_h, dx, COOLANT_NAME)
        P_c_profile[i-1] = P_c_local - dP

# ==============================================================================
# 6) Paper 곡선 PCHIP 보간 준비 (부드러운 비교용)
# ==============================================================================
def _pchip_sorted(xy, extrapolate=False):
    """(N,2) 배열을 x-정렬·중복 제거 후 PCHIP 생성"""
    xy = np.asarray(xy, dtype=float)
    xs, ys = xy[:,0], xy[:,1]
    s = np.argsort(xs)
    x_sorted, y_sorted = xs[s], ys[s]
    ux, idx = np.unique(x_sorted, return_index=True)
    uy = y_sorted[idx]
    return PchipInterpolator(ux, uy, extrapolate=extrapolate), (ux.min(), ux.max())

hf_pchip,  hf_rng  = _pchip_sorted(heat_flux_data, extrapolate=False)
Twg_pchip, Twg_rng = _pchip_sorted(T_wg_data,    extrapolate=False)
Tc_pchip,  Tc_rng  = _pchip_sorted(T_c_data,     extrapolate=False)
Pc_pchip,  Pc_rng  = _pchip_sorted(P_c_data,     extrapolate=False)

x_fine = np.linspace(-35, 25, 800)

def _safe_eval(pchip, rng, x):
    x0, x1 = rng
    mask = (x >= x0) & (x <= x1)
    y = np.full_like(x, np.nan, dtype=float)
    y[mask] = pchip(x[mask])
    return y, mask

# ==============================================================================
# 7) Plot (논문 Fig.6–8과 축/범위 일치, 평활 없음 + Paper PCHIP)
# ==============================================================================
x_cm = x_pos * 100.0
R_cm = (D_profile * 0.5) * 100.0

x_hf_ref,  hf_ref  = heat_flux_data[:,0], heat_flux_data[:,1]   # MW/m^2
x_Twg_ref, Twg_ref = T_wg_data[:,0],     T_wg_data[:,1]         # K
x_Tc_ref,  Tc_ref  = T_c_data[:,0],      T_c_data[:,1]          # K
x_Pc_ref,  Pc_ref  = P_c_data[:,0],      P_c_data[:,1]          # bar

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('1D Regenerative Cooling (Modified Bartz, frozen props, rib area)', fontsize=16)

# Heat flux
ax1_t = ax1.twinx()
ax1.plot(x_cm, q_profile/1e6, '-', color='g', label='Heat Flux')

y_hf, m_hf = _safe_eval(hf_pchip, hf_rng, x_fine)
ax1.plot(x_fine[m_hf], y_hf[m_hf], '--', color='purple', lw=1.6, label='Paper (PCHIP)')
ax1.plot(x_hf_ref, hf_ref, 'o', color='purple', ms=4, alpha=0.7)

ax1_t.plot(x_cm, R_cm, 'k', alpha=0.35, label='Engine Contour')
ax1.set_xlabel('Axial Position from the Throat (cm)')
ax1.set_ylabel('Heat Flux (MW/m^2)')
ax1_t.set_ylabel('Radial Position from the Nozzle Center Line (cm)')
ax1.set_xlim(-35, 25); ax1.set_ylim(0, 50); ax1_t.set_ylim(0, 60)
ax1.set_xticks(np.arange(-35, 26, 5)); ax1.grid(True)
h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax1_t.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper left')

# Gas-side wall temperature
ax2_t = ax2.twinx()
ax2.plot(x_cm, T_wg_profile, '-', color='r', label='Gas-side Wall Temp.')

y_Twg, m_Twg = _safe_eval(Twg_pchip, Twg_rng, x_fine)
ax2.plot(x_fine[m_Twg], y_Twg[m_Twg], '--', color='purple', lw=1.6, label='Paper (PCHIP)')
ax2.plot(x_Twg_ref, Twg_ref, 'o', color='purple', ms=4, alpha=0.7)

ax2_t.plot(x_cm, R_cm, 'k', alpha=0.35, label='Engine Contour')
ax2.set_xlabel('Axial Position from the Throat (cm)')
ax2.set_ylabel('Temperature (K)')
ax2_t.set_ylabel('Radial Position from the Nozzle Center Line (cm)')
ax2.set_xlim(-35, 25); ax2.set_ylim(0, 1500); ax2_t.set_ylim(0, 60)
ax2.set_xticks(np.arange(-35, 26, 5)); ax2.grid(True)
h1,l1 = ax2.get_legend_handles_labels(); h2,l2 = ax2_t.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc='upper left')

# Coolant temperature
ax3_t = ax3.twinx()
ax3.plot(x_cm, T_c_profile, '-', color='b', label='Coolant Bulk Temp.')

y_Tc, m_Tc = _safe_eval(Tc_pchip, Tc_rng, x_fine)
ax3.plot(x_fine[m_Tc], y_Tc[m_Tc], '--', color='purple', lw=1.6, label='Paper (PCHIP)')
ax3.plot(x_Tc_ref, Tc_ref, 'o', color='purple', ms=4, alpha=0.7)

ax3_t.plot(x_cm, R_cm, 'k', alpha=0.35, label='Engine Contour')
ax3.set_xlabel('Axial Position from the Throat (cm)')
ax3.set_ylabel('Coolant Temperature (K)')
ax3_t.set_ylabel('Radial Position from the Nozzle Center Line (cm)')
ax3.set_xlim(-35, 25); ax3.set_ylim(0, 600); ax3_t.set_ylim(0, 60)
ax3.set_xticks(np.arange(-35, 26, 5)); ax3.grid(True)
h1,l1 = ax3.get_legend_handles_labels(); h2,l2 = ax3_t.get_legend_handles_labels()
ax3.legend(h1+h2, l1+l2, loc='upper right')

# Coolant pressure
ax4_t = ax4.twinx()
ax4.plot(x_cm, P_c_profile/1e5, '-', color='c', label='Coolant Pressure')

y_Pc, m_Pc = _safe_eval(Pc_pchip, Pc_rng, x_fine)
ax4.plot(x_fine[m_Pc], y_Pc[m_Pc], '--', color='purple', lw=1.6, label='Paper (PCHIP)')
ax4.plot(x_Pc_ref, Pc_ref, 'o', color='purple', ms=4, alpha=0.7)

ax4_t.plot(x_cm, R_cm, 'k', alpha=0.35, label='Engine Contour')
ax4.set_xlabel('Axial Position from the Throat (cm)')
ax4.set_ylabel('Coolant Pressure (bar)')
ax4_t.set_ylabel('Radial Position from the Nozzle Center Line (cm)')
ax4.set_xlim(-35, 25); ax4.set_ylim(118, 132); ax4_t.set_ylim(0, 60)
ax4.set_xticks(np.arange(-35, 26, 5)); ax4.grid(True)
h1,l1 = ax4.get_legend_handles_labels(); h2,l2 = ax4_t.get_legend_handles_labels()
ax4.legend(h1+h2, l1+l2, loc='lower right')

for ax in (ax1, ax2, ax3, ax4):
    ax.axvline(0, color='grey', lw=1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

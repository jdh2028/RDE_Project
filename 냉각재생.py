import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

# ==============================================================================
# 1. 열 해석 함수 정의
# ==============================================================================
def bartz_h_g(Mach, T_wg, D, P_c, T_0, gamma, mu, Cp, Pr, w, C_star):
    T_aw = T_0 * (1 + Pr**(1/3) * (gamma - 1) / 2 * Mach**2)
    term1 = 0.5 * (T_wg / T_0) * (1 + (gamma - 1) / 2 * Mach**2)
    term2 = 0.5
    denominator = (term1 + term2)**(0.8 - w / 5) * (1 + (gamma - 1) / 2 * Mach**2)**(w / 5)
    if denominator == 0: return 1e5, T_aw
    sigma = 1 / denominator
    h_g = (0.026 / (D**0.2)) * ((mu**0.2 * Cp) / (Pr**0.6)) * ((P_c / C_star)**0.8) * sigma
    return h_g, T_aw

def qconv_h(T_aw, T_wg, h_g):
    return h_g * (T_aw - T_wg)

def calculate_alpha(m_dot_w, A_w, d, T_w, T_w_wall, pressure_Pa, fluid='Water'):
    try:
        if T_w_wall < 274 or T_w_wall > 600: return None
        cp_w = CP.PropsSI('C', 'T', T_w, 'P', pressure_Pa, fluid)
        mu_w = CP.PropsSI('V', 'T', T_w, 'P', pressure_Pa, fluid)
        k_w = CP.PropsSI('L', 'T', T_w, 'P', pressure_Pa, fluid)
        Pr_w = cp_w * mu_w / k_w
        cp_w_wall = CP.PropsSI('C', 'T', T_w_wall, 'P', pressure_Pa, fluid)
        mu_w_wall = CP.PropsSI('V', 'T', T_w_wall, 'P', pressure_Pa, fluid)
        k_w_wall = CP.PropsSI('L', 'T', T_w_wall, 'P', pressure_Pa, fluid)
        Pr_w_wall = cp_w_wall * mu_w_wall / k_w_wall
    except ValueError: return None
    Z = (k_w**0.57 * cp_w**0.43) / (mu_w**0.37)
    if Pr_w_wall == 0: return None
    Psi = (Pr_w / Pr_w_wall)**0.25
    mass_flux = m_dot_w / A_w
    alpha = (0.021 / (d**0.2)) * (mass_flux**0.8) * Z * Psi
    return alpha

def C_star(P_0, A_t, m_dot):
    return (P_0 * A_t) / m_dot

# ==============================================================================
# 2. 입력 변수 및 계산 준비
# ==============================================================================
# --- 연소 가스 물성치 (상수로 사용) ---
Pr_gas = 0.336
Cp_gas = 2289.3
T_0 = 2981.22    # K, 전 구간에서 이 온도를 사용
P_0 = 20e5
gamma = 1.2107
mu_gas = 7.78e-05
w = 0.6

# --- 기본 입력 ---
L_chamber = 0.3
D_chamber = 0.070
D_t = 0.0198
A_t = (np.pi * D_t**2) / 4
m_dot_gas = 0.45
Wall_Thick = 0.002
k_wall = 25.0
COOLANT_NAME = 'Water'
P_coolant = 25e5
m_dot_coolant = 1.0
channel_width = 2.0 * 1e-3
channel_height = 5.0 * 1e-3
num_channels = int(np.pi * D_chamber / (channel_width + Wall_Thick))
A_c_total = channel_width * channel_height * num_channels
d_h = 2 * (channel_width * channel_height) / (channel_width + channel_height)

# --- 계산 설정 ---
n_segments = 100
tol = 1e-4
max_iter = 1000
alpha_relax = 0.2
T_WG_INITIAL_GUESS = 900.0 # 논문과 유사한 값으로 시작
T_wg_guess_for_next_step = T_WG_INITIAL_GUESS 
# --- 초기 조건 ---
T_c_inlet = 291.7
C_star_val = C_star(P_0, A_t, m_dot_gas)
print(f"Calculated C* = {C_star_val:.2f} m/s")

x_pos = np.linspace(0, L_chamber, n_segments)
dx = x_pos[1] - x_pos[0]
D_profile = np.full(n_segments, D_chamber)
Mach_profile = np.linspace(0.01, 0.047, n_segments) # 입구에서 출구로 갈수록 Mach 증가

T_c_profile = np.zeros(n_segments)
T_wg_profile = np.zeros(n_segments)
T_wc_profile = np.zeros(n_segments)
q_profile = np.zeros(n_segments)
T_c_profile[-1] = T_c_inlet

# ==============================================================================
# 4. 메인 계산 루프
# ==============================================================================
print("\nStarting 1D thermal analysis based on correct interpretation...")

for i in reversed(range(n_segments)):
    D_local = D_profile[i]
    Mach_local = Mach_profile[i]
    T_c_local = T_c_profile[i]
    
    T_wg_iter = T_wg_guess_for_next_step
    
    for j in range(max_iter):
        # T_0를 직접 전달
        h_g, T_aw_local = bartz_h_g(Mach_local, T_wg_iter, D_local, P_0, T_0, gamma, mu_gas, Cp_gas, Pr_gas, w, C_star_val)
        q_conv = qconv_h(T_aw_local, T_wg_iter, h_g)

        T_wc_sub_iter = T_c_local + 50.0
        for k in range(20):
            h_c = calculate_alpha(m_dot_coolant, A_c_total, d_h, T_c_local, T_wc_sub_iter, P_coolant, COOLANT_NAME)
            if h_c is None: h_c = 30000.0
            T_wc_new_sub = T_c_local + q_conv / h_c
            if abs(T_wc_new_sub - T_wc_sub_iter) < 0.1: break
            T_wc_sub_iter = T_wc_new_sub
        T_wc_final = T_wc_sub_iter
                
        T_wg_new = T_wc_final + q_conv * Wall_Thick / k_wall
        
        if abs(T_wg_new - T_wg_iter) < tol: break
        T_wg_iter = T_wg_iter * (1 - alpha_relax) + T_wg_new * alpha_relax
    
    if j == max_iter - 1: print(f"Warning: Convergence not reached at segment {i}")

    T_wg_profile[i] = T_wg_new
    T_wc_profile[i] = T_wc_final
    q_profile[i] = q_conv
    
    T_wg_guess_for_next_step = T_wg_new

    if i > 0:
        A_heat_transfer = np.pi * D_local * dx
        Q_total = q_conv * A_heat_transfer
        try:
            Cp_c = CP.PropsSI('C', 'T', T_c_local, 'P', P_coolant, COOLANT_NAME)
            delta_T_c = Q_total / (m_dot_coolant * Cp_c)
            T_c_profile[i-1] = T_c_local + delta_T_c
        except ValueError:
            T_c_profile[i-1] = T_c_local

print("Calculation finished.")
print(f"Coolant Outlet Temperature: {T_c_profile[-1]:.2f} K ({T_c_profile[-1]-273.15:.2f} C)")
print(f"Max Hot-Gas Wall Temperature: {np.max(T_wg_profile):.2f} K")

# ==============================================================================
# 5. 결과 시각화
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:red'
ax1.set_xlabel('Position along chamber (m)')
ax1.set_ylabel('Temperature (K)', color=color)
ax1.plot(x_pos, T_wg_profile, label='Gas-side Wall Temp (T_wg)', color='red', linestyle='-')
ax1.plot(x_pos, T_wc_profile, label='Coolant-side Wall Temp (T_wc)', color='orange', linestyle='--')
ax1.plot(x_pos, T_c_profile, label='Coolant Temp (T_c)', color='blue', linestyle=':')

ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)
ax1.set_ylim(0, 3500)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Heat Flux (MW/m^2)', color=color)
ax2.plot(x_pos, q_profile / 1e6, label='Heat Flux (q)', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 100)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best')

fig.tight_layout()
plt.title('1D Thermal Analysis with T_gas(x) Profile')
plt.show()
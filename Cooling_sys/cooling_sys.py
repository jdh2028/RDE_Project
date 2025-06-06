#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt

import numpy as np

import CoolProp.CoolProp as CP

# In[2]:


def bartz_h_g(Mach, T_wg, D, P_c, T_0, gamma, mu, Cp, Pr, w, C_star):
    sigma = 1 / ((0.5*(T_wg/T_0)*(1+(gamma-1)/2*Mach**2)+(gamma+1)/2)**(0.8-w/5) * (1+(gamma-1)/2*Mach**2)**(w/5))
    h_g = (0.026 / (D**0.2)) * ((mu**0.2)*(Cp/Pr**0.6)) * ((P_c/C_star)**0.8) *  sigma
    return h_g


# In[3]:


# h_g : bartz 식을 통한 가스 열 전달 계수
# w = 물성지수 일반적으로 w = 0.6 사용
#D_t = 목 직경 , R = 곡률반지름


#곡률 반경을 고려한 바츠 eqn , C-D노즐에서 사용가능능
"""
def bartz_h_g(Mach, T_wg, D, P_c, T_0, gamma, mu, Cp, Pr, w, C_star, D_t ,R , A_t , A):
    sigma = 1 / ((0.5*(T_wg/T_0)*(1+(gamma-1)/2*Mach**2)+(gamma+1)/2)**(0.8-w/5) * (1+(gamma-1)/2*Mach**2)**w/5)
    h_g = (0.026 / (D**0.2)) * ((mu**0.2)*(Cp/Pr**0.6)) * ((P_c/C_star)**0.8)*(D_t/R)**0.1*(A_t/A) * sigma
    return h_g
"""


# In[4]:


# h_c : 냉각제의 열전달계수
# k_c (W/m*k) : 냉각제의 열전도율 , D_c 냉각로 직경 , Re_c Pr_c : 냉각제 관한 레이놀즈수, 프란틀 수 


def h_cf(k_c, D_c, Re_c, Pr_c):
    
    
    h_c = 0.023 * k_c / D_c * Re_c **0.8 * Pr_c **0.4

    return h_c


# In[5]:


# Coolant에 관한 레이놀즈수, 프란틀 수 뽑아줌
# Cp_c : 냉각제 정압비열, mu_c : 냉각제 점성계수 rho_c : 냉각제 밀도 , u_c : 냉각제 속도도
def Coolant_cal(k_c , Cp_c, mu_c, rho_c, u_c, D_c):

    Re_c = (rho_c * u_c * D_c)/ mu_c

    Pr_c = (Cp_c * mu_c)/ k_c

    return Re_c , Pr_c


# In[6]:


# T_aw : 열 손실이 전혀 없는 벽에 고온 유동이 접했을 때, 벽 표면이 도달하게 되는 유동의 경계층 온도

def T_aw(Pr_gas, Mach_gas, T_0, gamma):

  T_aw = T_0 * (1 + (Pr_gas**(1/3)) * ((gamma - 1) / 2) * (Mach_gas**2)) / (1 + ((gamma - 1) / 2) * (Mach_gas**2))

  return T_aw


# In[7]:


# T_wg : 가스 쪽 벽면 온도

def T_wgf(q_conv, T_wc, Wall_Thick, K_w):
  T_wg = T_wc + (Wall_Thick * q_conv /  K_w)
  return T_wg

def T_wg_out(T_wc, wall_thick, q_conv, k_w):
   
   T_wg2 = T_wc + q_conv*wall_thick/k_w

   return T_wg2


def T_wc_in(T_wg, wall_thick, q_conv, k_w):
   
   T_wc1 = T_wg - (wall_thick * q_conv / k_w)

   return T_wc1


def qconv_h(T_aw, T_wg, h_g):

  qconv_h = h_g* (T_aw -T_wg)

  return qconv_h



# In[9]:


#특성 속도
def C_star(P_0 , A_t , m_dot):

    

    return (P_0 * A_t) / m_dot


# In[10]:


# 기초 설정(가스 성분)


# 가스 관련 성분
Pr_gas = 0.8     #가스의 Pr계수
Mach_gas = 1     #가스의 마하수
Cp_gas = 2289.3     #가스의 정압비열
T_0 = 2981.22     #가스의 전온도 [K]
P_0 = 20e5        #가스의 전압력 [pa]
Gamma = 1.2107      #가스의 비열비 
M_dot = 0.45      #가스의 질유량
Mu = 7.78e-05         #가스의 점성계수
w = 0.6          #물성지수성지수



#구조 성분

Wall_Thick = 0.5*1e-3 #[m]


k_w = 120             #벽 열전도율

#area = 1.398e-3     #[m^2]
D_t = 0.0198 # 목 직경
A_t = (np.pi * D_t**2)/4    # 목 면적
R = 0             #곡률반지름



D_c = 2e-3          # 냉각유로 직경[m]
A_c = (np.pi * D_c**2)/4


#냉각제 성분



COOLANT_NAME = 'Water'
P_coolant = 20e5

M_dot_c = 0.1     # 냉각제 질량유량 [kg/s]






#초기 설정값(실행관련 성분)


tol = 1e-3          #열전달 계수 오차
max_iter = 1000      #열 전달계수 반복 계산
alpha = 0.2         #수렴 계수




T_c_init = 291.7     # 냉각제 입구 온도 [K]
T_wg_init = 500     # 가스측 벽 온도 초기값

T_c = T_c_init
T_wg = T_wg_init
T_0_local = T_0








# 수렴 기반 반복 계산
T_wg_arr = []
T_wc_arr = []
q_arr = []
T_c_arr = []
T_aw_arr = []


C_star_val  = C_star(P_0 , A_t , M_dot)

x = np.linspace(0, 0.3, 100)
A_x = A_t * np.ones_like(x)



# dx 계산 (x는 축방향 위치 배열)
dx = x[1] - x[0]

Coolant_Area = D_c * np.pi /2 * dx # 냉각제가 만나는 열과 만나는 면적적



for i, oi in enumerate(x):
    

    h_g = 0.0
    q = 0.0
    T_c1 = 0.0
    T_wc = 0.0

    rho_c = CP.PropsSI('D', 'T', T_c, 'P', P_coolant, COOLANT_NAME)
    mu_c  = CP.PropsSI('V', 'T', T_c, 'P', P_coolant, COOLANT_NAME)
    Cp_c  = CP.PropsSI('Cpmass', 'T', T_c, 'P', P_coolant, COOLANT_NAME)
    k_c   = CP.PropsSI('L', 'T', T_c, 'P', P_coolant, COOLANT_NAME)

    U_c_local = M_dot_c / (rho_c * A_c)

    
    
    Re_c_local, Pr_c_local = Coolant_cal(k_c, Cp_c, mu_c, rho_c, U_c_local, D_c)
    h_c = h_cf(k_c, D_c, Re_c_local, Pr_c_local)

    T_aw_val = T_aw(Pr_gas, Mach_gas, T_0_local, Gamma)

    for _ in range(max_iter):
        
        h_g = bartz_h_g(Mach_gas, T_wg, D_t, P_0, T_0_local, Gamma, Mu, Cp_gas, Pr_gas, w, C_star_val)              
        q = qconv_h(T_aw_val, T_wg, h_g)
        Q = q * Coolant_Area
        T_c1 = Q /(M_dot_c*Cp_c) + T_c
        T_wc = T_c1 + q / h_c
        T_wg_new = T_wgf(q, T_wc, Wall_Thick, k_w)
        
        
        if abs(T_wg_new - T_wg) < tol:
            
            break
        
        T_wg = (1-alpha)*T_wg + alpha*T_wg_new
    
    h_g_final = bartz_h_g(Mach_gas, T_wg_new, D_t, P_0, T_0_local, Gamma, Mu, Cp_gas, Pr_gas, w, C_star_val)
    q_final = qconv_h(T_aw_val, T_wg_new, h_g_final)
    

    T_c = T_c1
   
    delta_T_0_gas = - (q_final * np.pi * D_t * dx) / (M_dot * Cp_gas) 
    T_0_gas_local_new = T_0_local + delta_T_0_gas
    T_0_local = T_0_gas_local_new

    # 저장
    T_wg_arr.append(T_wg_new)
    T_wc_arr.append(T_wc)
    q_arr.append(q_final)
    T_c_arr.append(T_c1)
    T_aw_arr.append(T_aw_val)



# 온도 플롯
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(x, T_wg_arr, label='T_wg (Gas-side Wall Temp)', linewidth=2, color='black')
ax1.plot(x, T_c_arr, label='T_c (Coolant Temp)', linewidth=2, color='blue')
ax1.plot(x, T_aw_arr, label='T_aw (Gas Temp)', linewidth=4, color='red')

ax1.set_xlabel('Nozzle Axial Position x [m]', fontsize=12)
ax1.set_ylabel('Temperature [K]', fontsize=12)
ax1.set_ylim(bottom=0)
ax1.grid(True)

# 노즐 형상 (하단 표시)
ax1.fill_between(x, 0, A_x / np.max(A_x) * np.min(T_c_arr) * 0.3,
                 color='gray', alpha=0.3, label='Nozzle Shape (scaled)')

# 오른쪽 y축: 열유속
ax2 = ax1.twinx()
ax2.plot(x, q_arr, label='Heat Flux q', color='orange', linewidth=1.5)
ax2.set_ylabel('Heat Flux [W/m²]', fontsize=12, color='orange')
ax2.tick_params(axis='y', labelcolor='orange')


ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.65))
ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.85))  


plt.title('Temperature and Heat Flux Distribution Along 1-D Nozzle', fontsize=14)
plt.tight_layout()
plt.show()





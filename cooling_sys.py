#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt

import numpy as np


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
D_t = 0.3 # 목 직경
A_t = (np.pi * D_t**2)/4    # 목 면적
R = 0             #곡률반지름



D_c = 2e-3          # 냉각유로 직경[m]


#냉각제 성분

#T_c = 291.7           # 냉각제 입구 온도 (초기)
k_c = 0.6            # 냉각제 열전도율

Cp_c = 2289         # 냉각제 정압비열  [J/kg·K]
Mu_c = 20.869e-6          # 냉각제 점성계수
Rho_c = 958.4         # 냉각제 밀도
U_c = 30            # 냉각제 속도
M_dot_c = U_c * Rho_c * (np.pi * D_c**2)/4     # 냉각제 질량유량 [kg/s]


# In[ ]:


print(M_dot_c)


# In[18]:


#초기 설정값(실행관련 성분)

         #[K] 초기 설정값

tol = 1e-3          #열전달 계수 오차
max_iter = 1000      #열 전달계수 반복 계산
alpha = 0.2         #수렴 계수


# In[ ]:


T_c_init = 291.7     # 냉각제 입구 온도 [K]
T_wg_init = 500     # 가스측 벽 온도 초기값

T_c = T_c_init
T_wg = T_wg_init


T_aw_val = T_aw(Pr_gas, Mach_gas, T_0, Gamma)

Re_c, Pr_c = Coolant_cal(k_c, Cp_c, Mu_c, Rho_c, U_c, D_c)
h_c = h_cf(k_c, D_c, Re_c, Pr_c)




# 수렴 기반 반복 계산
T_wg_arr = []
T_wc_arr = []
q_arr = []
T_c_arr = []



C_star_val  = C_star(P_0 , A_t , M_dot)

x = np.linspace(0, 0.3, 100)
A_x = A_t * np.ones_like(x)



# dx 계산 (x는 축방향 위치 배열)
dx = x[1] - x[0]

area =  D_c * dx

T_aw_val = T_aw(Pr_gas, Mach_gas, T_0, Gamma)

for i, oi in enumerate(x):
    
   
    for _ in range(max_iter):
        
        h_g = bartz_h_g(Mach_gas, T_wg, D_t, P_0, T_0, Gamma, Mu, Cp_gas, Pr_gas, w, C_star_val)              
        q = qconv_h(T_aw_val, T_wg, h_g)
        
        T_c1 = q/(M_dot_c*Cp_c) + T_c
        T_wc = T_c1 + q / h_c
        T_wg_new = T_wgf(q, T_wc, Wall_Thick, k_w)
        
        
        if abs(T_wg_new - T_wg) < tol:
            
            break
        
        T_wg = (1-alpha)*T_wg + alpha*T_wg_new
        
    


    
    
    

    # 저장
    T_wg_arr.append(T_wg_new)
    T_wc_arr.append(T_wc)
    q_arr.append(q)
    T_c_arr.append(T_c1)




# 온도 플롯
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(x, T_wg_arr[:], label='T_wg (Gas-side Wall Temp)', linewidth=2, color='red')
ax1.plot(x, T_wc_arr[:], label='T_wc (Coolant-side Wall Temp)', linewidth=2, color='blue')
ax1.plot(x, T_c_arr[:], label='T_c (Coolant Temp)', linestyle='--', linewidth=2, color='green')

ax1.set_xlabel('Nozzle Axial Position x [m]', fontsize=12)
ax1.set_ylabel('Temperature [K]', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(True)
plt.title('Temperature Distribution Along Nozzle', fontsize=14)
plt.tight_layout()
plt.show()





import numpy as np
import time
import matplotlib.pyplot as plt

print("Gerando dados de telemetria...")
X_f64 = np.random.rand(6000000, 5)
X_i8 = (X_f64 * 127).astype(np.int8)

W_f64 = np.random.rand(5, 10)
W_i8 = (W_f64 * 127).astype(np.int8)

print("Processando Float64...")
start_time = time.time()
Z_f64 = np.dot(X_f64[:1000000, :], W_f64)
tempo_f64 = (time.time() - start_time) / 60

print("Processando Int8...")
start_time = time.time()
Z_i8 = np.dot(X_i8[:1000000, :].astype(np.int32), W_i8.astype(np.int32))
tempo_i8 = (time.time() - start_time) / 60

print(f"Tempo Float64: {tempo_f64:.4f} min")
print(f"Tempo Int8: {tempo_i8:.4f} min")

categorias = ['int8', 'float64']
tempos = [tempo_i8, tempo_f64]

plt.bar(categorias, tempos, color=['green', 'blue'])
plt.ylabel('Tempo de Processamento (Minutos)')
plt.title('Impacto do Tipo de Dado no Tempo de Execução (MLP - 64 bits)')
plt.savefig('resultado_execucao.png')
print("Gráfico 'resultado_execucao.png' salvo com sucesso!")
import numpy as np
import time

def sigmoide(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

print("=== INICIANDO PIPELINE DE TELEMETRIA DE POUSO (MLP) ===")

num_linhas = 6000000
num_atributos = 5
print(f"Gerando dataset sintético global: {num_linhas} linhas e {num_atributos} colunas...")

X_global = np.random.randn(num_linhas, num_atributos).astype(np.float64)

y_global = np.random.uniform(0, 1, (num_linhas, 1)).astype(np.float64)

tamanho_lote = 1000000
X_lote = X_global[:tamanho_lote, :]
y_lote = y_global[:tamanho_lote, :]
print(f"Lote fatiado para processamento embarcado: {X_lote.shape}\n")

W_oculta_f64 = np.random.randn(num_atributos, 10).astype(np.float64)
W_saida_f64 = np.random.randn(10, 1).astype(np.float64)

print("Executando o mecanismo de quantização dos dados para Int8...")
X_lote_int8 = np.trunc(X_lote * 127).astype(np.int8)
W_oculta_int8 = np.trunc(W_oculta_f64 * 127).astype(np.int8)
W_saida_int8 = np.trunc(W_saida_f64 * 127).astype(np.int8)

epocas = 10

print("\nIniciando treinamento iterativo em Float64...")
inicio_f64 = time.time()

for epoca in range(epocas):
    z_oculta = np.dot(X_lote, W_oculta_f64)
    a_oculta = sigmoide(z_oculta)

    z_saida = np.dot(a_oculta, W_saida_f64)
    a_saida = sigmoide(z_saida)

erro_mse_f64 = np.mean((y_lote - a_saida) ** 2)

fim_f64 = time.time()
tempo_f64_minutos = (fim_f64 - inicio_f64) / 60

print(f">> Resultado Float64 -> Épocas: {epocas} | Erro Final (MSE): {erro_mse_f64:.4f} | Tempo: {tempo_f64_minutos:.5f} minutos")

print("\nIniciando treinamento iterativo em Int8 (Com conversão em tempo de execução)...")
inicio_int8 = time.time()
for epoca in range(epocas):
    X_f64_convertido = X_lote_int8.astype(np.float64) / 127.0
    W_oc_f64_convertido = W_oculta_int8.astype(np.float64) / 127.0
    W_sc_f64_convertido = W_saida_int8.astype(np.float64) / 127.0

    z_oculta = np.dot(X_f64_convertido, W_oc_f64_convertido)
    a_oculta = sigmoide(z_oculta)

    z_saida = np.dot(a_oculta, W_sc_f64_convertido)
    a_saida = sigmoide(z_saida)

    erro_mse_int8 = np.mean((y_lote - a_saida) ** 2)

    fim_int8 = time.time()
    tempo_int8_minutos = (fim_int8 - inicio_int8) / 60

    print( f">> Resultado Int8 -> Épocas: {epocas} | Erro Final (MSE): "
        f"{erro_mse_int8:.4f} | Tempo: {tempo_int8_minutos:.5f} minutos")


print("\n=== PIPELINE CONCLUÍDO COM SUCESSO ===")


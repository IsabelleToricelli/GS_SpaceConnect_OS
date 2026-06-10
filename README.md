# Simulação de Rede Neural (MLP) para Computação de Borda Aeroespacial 

## Objetivo do Projeto
Este projeto foi desenvolvido com o objetivo de simular uma rede neural em um ambiente de computação de borda aeroespacial. No espaço, o computador de bordo precisa ser extremamente compacto e, por conta disso, possui restrições severas como pouca memória, bateria limitada dependente de painéis solares e conectividade de rede muito lenta. 

Para lidar com essas limitações, o sistema avalia qual o melhor formato para processar 6 milhões de dados de telemetria. A primeira opção analisada é o formato padrão dos computadores, o **Float64**, em que os números são mais precisos, porém cada elemento ocupa 8 bytes, resultando em um armazenamento pesado para o cenário de satélites. A segunda opção é o formato **Int8** (obtido por meio de quantização), onde o dado passa a ocupar apenas 1 byte.

A simulação da rede neural compara o tempo de execução e a eficiência de ambos os formatos, ajudando a compreender o melhor caminho para atingir o equilíbrio ideal entre precisão matemática e economia de hardware no espaço.

---

##  Especificação Arquitetural da Rede Neural
O funcionamento da arquitetura da rede neural baseia-se em um fluxo direto (*forward pass*), no qual são recebidos cinco dados de sensores do satélite. A camada de entrada processa cinco informações físicas em tempo real:

* Altura do satélite (**Altitude**)
* Velocidade do satélite no espaço (**Velocidade Orbital**)
* **Pressão** em volta
* **Temperatura**
* Ângulo em que o satélite está inclinado (**Inclinação**)

Esses dados são enviados para uma camada oculta composta por **10 neurônios**. Cada um deles possui a função de receber esses cinco dados de entrada e multiplicá-los por pesos, que funcionam como o nível de importância que cada neurônio dá para cada informação recebida. 

Para realizar essa operação de forma rápida e leve, os neurônios utilizam uma função de ativação **Linear**. O sistema pega a matriz com os dados dos sensores ($X$) e a multiplica diretamente pela matriz de pesos dos neurônios ($W$), seguindo a fórmula de produto escalar:

$$Z = X \cdot W$$

---

## Pipeline de Processamento de Dados
O fluxo do processamento foi dividido em três etapas principais dentro do sistema geral:

1. **Geração dos Dados:** O sistema cria uma matriz global com 6 milhões de linhas e 5 colunas (uma para cada sensor). Os números gerados são aleatórios e normalizados (na mesma escala entre 0.0 e 1.0), simulando o histórico de telemetria gerado pelo satélite.
2. **Fatiamento do Lote (Buffer):** Como o computador do satélite possui limitações de hardware, ele não seria capaz de processar as 6 milhões de linhas de uma só vez. Por isso, o código faz um fatiamento, analisando apenas o primeiro lote crítico de 1 milhão de linhas da tabela original, mantendo as 5 colunas intactas.
3. **Mecanismo de Quantização:** Como os dados do lote estão em formato decimal pesado (Float64), a rede neural transforma esses dados em inteiros leves (Int8) multiplicando cada número por 127 e cortando as casas decimais. Essa ação reduz significativamente o tamanho do arquivo na memória sem corromper as informações dos sensores.

---

## Análise de Resultados e Conclusão
O sistema gera um gráfico de barras comparando o tempo que o processador levou para realizar os cálculos com os dois formatos:

* **Float64 (Barra Azul):** ~0.0003 minutos
* **Int8 (Barra Verde):** ~0.0014 minutos

### Justificativa Técnica
Os cálculos usando o formato **Float64** demandaram cerca

<!-- @format -->

<img src="./assets/header.png"/>

<h1 align="center">TIAMAT</h1>

<p align="center">
  <img src="./assets/Samurais de Takemikazuchi.png" alt="Header" width="200"/>
</p>

<p align="center">
    <i>Modelagem de sinais sísmicos utilizando técnicas de Aprendizado de Máquina</i>
    <br>
    <i>ILUM - Escola de Ciência | Campinas, 2026</i>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white"/>
    <img src="https://img.shields.io/badge/Aprendizado%20de%20Máquina-ML-orange"/>
    <img src="https://img.shields.io/badge/Ciência%20de%20Dados-Data%20Science-green"/>
    <img src="https://img.shields.io/badge/Seismologia-Geofísica-purple"/>
</p>

---

## Sobre o repositório

Esse é um projeto desenvolvido para a disciplina de **Aprendizado de Máquina**, com o objetivo de investigar a aplicação de técnicas de Machine Learning na análise de sinais associados a eventos sísmicos observados no contexto do Sirius, no **CNPEM**.

O projeto combina informações provenientes de catálogos sísmicos, registros de eventos e dados experimentais obtidos por instrumentos do Sirius. A partir desses dados, são calculadas características relacionadas à localização e à propagação das ondas sísmicas, que posteriormente são utilizadas na construção de modelos preditivos.

O desenvolvimento do projeto foi dividido em duas etapas principais:

- **Coleta e Tratamento de Dados**
- **Modelagem utilizando Aprendizado de Máquina**

---

## Conteúdos do repositório

<ul>
  <li> 📄 <b>README.md</b>: documento principal com a apresentação, metodologia e organização do projeto</li>

  <li> 📂 <b>00 - Coleta e Tratamento de Dados</b>: contém os notebooks e scripts utilizados para a obtenção, processamento e organização dos dados sísmicos</li>
  <ul>
    <li> 📓 <b>01 - Coleta de Logs.ipynb</b>: coleta e organização dos registros de eventos sísmicos associados aos logs do Sirius</li>
    <li> 📓 <b>02 - Coleta de PVs.ipynb</b>: coleta dos sinais provenientes dos instrumentos de monitoramento do Sirius</li>
    <li> 📓 <b>03 - Coleta de Terremotos.ipynb</b>: consulta a catálogos sísmicos e obtenção das informações dos terremotos</li>
    <li> 📓 <b>04 - Cálculo de Ondas de Superfície.ipynb</b>: cálculo dos tempos de propagação das ondas sísmicas de superfície</li>
    <li> 📓 <b>05 - Cálculo de Ondas de Corpo.ipynb</b>: cálculo dos tempos de propagação das ondas sísmicas de corpo</li>
    <li> 📓 <b>06 - Cálculo de Chegada.ipynb</b>: determinação dos tempos esperados de chegada dos eventos sísmicos aos instrumentos</li>
    <li> 📓 <b>07 - Análise e Tratamento.ipynb</b>: análise, processamento e preparação dos dados para a etapa de modelagem</li>
    <li> 📂 <b>Dados</b>: contém os datasets e arquivos utilizados durante o desenvolvimento do projeto</li>
  <ul>
    <li> 📊 <b>Dataset final</b>: conjunto de dados utilizado na etapa de modelagem</li>
    <li> 📊 <b>Dados sísmicos</b>: informações provenientes dos catálogos de terremotos e dos instrumentos do Sirius</li>
    <li> 📊 <b>Dados processados</b>: sinais e informações obtidas após as etapas de tratamento</li>
  </ul>

  <li> 📂 <b>Scripts</b>: contém funções auxiliares utilizadas na coleta, processamento, análise e visualização dos dados</li>
  <ul>
    <li> 🐍 <b>Funções de coleta</b>: ferramentas para obtenção dos dados dos catálogos e instrumentos</li>
    <li> 🐍 <b>Funções sísmicas</b>: cálculos relacionados à propagação e chegada das ondas</li>
    <li> 🐍 <b>Funções de análise</b>: ferramentas para processamento e visualização dos sinais</li>
  </ul>
  </ul>

  <li> 📂 <b>01 - Modelos</b>: contém os notebooks utilizados para a construção, otimização e avaliação dos modelos de Machine Learning</li>
  <ul>
    <li> 📓 <b>01.A - Baseline.ipynb</b>: construção de um modelo de referência para comparação com os demais algoritmos</li>
<li> 📓 <b>01.B - KNN.ipynb</b>: aplicação e otimização do algoritmo K-Nearest Neighbors</li>
<li> 📓 <b>01.C - Árvore de Decisão.ipynb</b>: aplicação e otimização de Árvores de Decisão</li>
<li> 📓 <b>01.D - Floresta Aleatória.ipynb</b>: aplicação e otimização do algoritmo de Floresta Aleatória</li>
<li> 📓 <b>01.E - Gradient Boosting.ipynb</b>: aplicação e otimização do algoritmo Gradient Boosting</li>
<li> 📓 <b>01.F - XGBoost.ipynb</b>: aplicação e otimização do algoritmo XGBoost</li>
      <li> 📂 <b>Studies</b>: contém os estudos do Optuna utilizados na otimização dos hiperparâmetros</li>
  <ul>
    <li> 🗃️ <b>knn_tiamat.db</b>: estudo de otimização do modelo KNN</li>
    <li> 🗃️ <b>dt_tiamat.db</b>: estudo de otimização da Árvore de Decisão</li>
    <li> 🗃️ <b>rf_tiamat.db</b>: estudo de otimização da Floresta Aleatória</li>
  </ul>
  </ul>

  <li> 📂 <b>assets</b>: contém os recursos visuais utilizados no README e no projeto</li>

  <li> 📄 <b>LICENSE</b>: licença de uso e distribuição do projeto</li>
</ul>

---

## Metodologia

O desenvolvimento do TIAMAT segue um fluxo dividido em diferentes etapas.

### 1. Coleta dos dados

Inicialmente, são identificados eventos sísmicos associados aos registros disponíveis no ambiente do Sirius. Esses eventos são complementados por informações provenientes de catálogos sísmicos, permitindo obter características como:

- magnitude;
- profundidade;
- latitude e longitude;
- tipo de magnitude;
- distância até o Sirius;
- azimute;
- localização tectônica;
- proporção entre percurso sobre terra e água.

Além das informações do evento, são coletados os sinais registrados pelos instrumentos de monitoramento.

### 2. Processamento dos sinais

Os sinais sísmicos são analisados para identificar características associadas à chegada das ondas.

São consideradas as componentes dos sinais registradas pelos instrumentos, permitindo analisar a resposta do sistema diante de diferentes eventos sísmicos.

### 3. Cálculo da propagação das ondas

A partir da localização do evento e do Sirius, são calculados os tempos esperados de chegada de diferentes tipos de ondas sísmicas.

Esses valores permitem comparar o comportamento previsto pela física com o comportamento efetivamente observado nos sinais.

### 4. Construção do dataset

As informações obtidas nas etapas anteriores são combinadas em um dataset estruturado.

Entre as variáveis utilizadas estão características relacionadas ao evento sísmico, à geometria entre o terremoto e o Sirius e aos tempos de propagação das ondas.

O dataset constitui a base para a etapa de modelagem.

### 5. Modelagem

Foram avaliados diferentes algoritmos de regressão:

- **Baseline**, utilizado como referência;
- **K-Nearest Neighbors (KNN)**;
- **Árvore de Decisão**;
- **Floresta Aleatória (Random Forest)**.

A seleção dos hiperparâmetros dos modelos é realizada utilizando **Optuna**, permitindo explorar diferentes configurações e comparar seus desempenhos.

---

## Modelos utilizados

### Baseline

O modelo baseline estabelece uma referência para avaliar se os modelos de Machine Learning conseguem obter desempenho superior a uma estratégia simples de predição.

### K-Nearest Neighbors

O KNN realiza previsões a partir das observações mais próximas no espaço de atributos.

Foram explorados diferentes valores de:

- número de vizinhos;
- função de ponderação dos vizinhos;
- métrica de distância de Minkowski.

### Árvore de Decisão

A Árvore de Decisão realiza sucessivas divisões do espaço de atributos para produzir uma previsão.

Entre os hiperparâmetros explorados estão:

- profundidade máxima;
- número mínimo de amostras para divisão;
- número mínimo de amostras por folha;
- quantidade de atributos considerados em cada divisão.

### Floresta Aleatória

A Floresta Aleatória combina diversas árvores de decisão para produzir uma previsão mais robusta.

Foram avaliados diferentes valores para:

- número de árvores;
- profundidade máxima;
- número mínimo de amostras para divisão;
- número mínimo de amostras por folha;
- quantidade de atributos considerados em cada divisão.

### Gradient Boosting

O Gradient Boosting combina diversas árvores de decisão construídas sequencialmente, em que cada nova árvore busca corrigir os erros das árvores anteriores.

Foram explorados diferentes valores de:

- número de árvores;
- taxa de aprendizado;
- profundidade máxima;
- número mínimo de amostras para divisão;
- número mínimo de amostras por folha.

### XGBoost

O XGBoost é uma implementação otimizada do Gradient Boosting, utilizando árvores de decisão construídas sequencialmente para melhorar as previsões do modelo.

Foram avaliados diferentes valores para:

- número de árvores;
- taxa de aprendizado;
- profundidade máxima;
- peso mínimo necessário para realizar uma divisão;
- fração das amostras utilizadas em cada árvore;
- fração dos atributos considerados em cada árvore.

---

## Otimização dos hiperparâmetros

A otimização dos modelos foi realizada utilizando a biblioteca **Optuna**.

Os estudos são armazenados em bancos de dados SQLite, permitindo preservar os experimentos e continuar as buscas posteriormente.

Cada algoritmo possui seu próprio estudo:

```text
Studies/
├── knn_tiamat.db
├── dt_tiamat.db
└── rf_tiamat.db
```

A avaliação dos modelos é realizada utilizando validação cruzada, enquanto os hiperparâmetros são selecionados com base no desempenho obtido durante os experimentos.

---

## Tecnologias utilizadas

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white"/>
    <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white"/>
    <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white"/>
    <img src="https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white"/>
    <img src="https://img.shields.io/badge/Optuna-ML%20Optimization-6A5ACD"/>
    <img src="https://img.shields.io/badge/ObsPy-Seismology-orange"/>
</p>

Entre as principais ferramentas utilizadas estão:

- **Python**;
- **NumPy**;
- **Pandas**;
- **Scikit-learn**;
- **Optuna**;
- **ObsPy**;
- **Matplotlib**;
- **Seaborn**;
- **Plotly**.

---

## Descrição do uso de IA neste trabalho

Durante o desenvolvimento do projeto, ferramentas de Inteligência Artificial foram utilizadas como apoio ao processo de desenvolvimento e aprendizagem.

A IA foi utilizada principalmente para:

- auxiliar na identificação e correção de erros de programação;
- discutir estratégias de processamento dos sinais;
- explorar possibilidades de engenharia de atributos;
- auxiliar na interpretação dos resultados dos modelos;
- discutir alternativas de validação e avaliação;
- esclarecer conceitos relacionados a Machine Learning, processamento de sinais e análise de dados.

As decisões metodológicas, implementação final, análise dos resultados e interpretação científica foram realizadas pelos autores do projeto.

---

## Agradecimentos

Agradecemos aos colaboradores e professores envolvidos no desenvolvimento do projeto e nas atividades da disciplina de **Aprendizado de Máquina**, especialmente ao Afonso Haruo e ao Wagner Luiz da sala de controle do Sirius que nos ajudaram a obter os dados necessários para desenveolver o projeto.

Também agradeço ao meu colega Vinícius Marianno de Marque Cutolo por ter auxiliado na etapa de processamento dos sinais sísmicos

---

## XKCD Relevante

<img src="https://imgs.xkcd.com/comics/seismic_waves.png"></img>

`Imagem: Seismic Waves (XKCD) disponível em https://xkcd.com/723`

---

## Professor orientador

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/drcassar">
        <img src="https://avatars.githubusercontent.com/u/9871905?v=4" width="100px;" alt="Foto do Daniel no Github" style="border-radius:100%;"/><br>
        <b>Prof. Dr. Daniel R. Cassar</b>
      </a>
    </td>
  </tr>
</table>

---

## Autor

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/FipaLipe">
        <img src="https://avatars.githubusercontent.com/u/90717547?v=4" width="100px;" alt="Foto do Filipi no Github" style="border-radius:100%;"/><br>
        <b>Filipi Martins</b>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/ViniCutolo">
        <img src="https://avatars.githubusercontent.com/u/271505502?v=4" width="100px;" alt="Photo of Vinícius" style="border-radius:100%;"/><br>
        <b>Vinícius Cutolo</b>
      </a>
    </td>
  </tr>
</table>

---

<p align="center">
    <i>TIAMAT — Segunda entrega da disciplina de Aprendizado de Máquina</i>
</p>

# 🧩 Sistema de Análise de Autismo no Brasil

Este projeto é uma aplicação de análise de dados aplicada à saúde, especificamente focada no autismo no Brasil. O sistema utiliza uma base de dados sintética gerada por algoritmos de machine learning para explorar características, padrões e realizar previsões relacionadas ao autismo no contexto brasileiro.

## 📋 Tabela de Conteúdo
- [Visão Geral](#visão-geral)
- [Funcionalidades](#funcionalidades)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Desenvolvedores](#desenvolvedores)

## 🔍 Visão Geral

O sistema analisa e simula dados relacionados ao autismo no Brasil, permitindo visualizar tendências, padrões e fazer previsões com base em diferentes variáveis. A aplicação utiliza modelos de machine learning para prever informações relevantes, como número de consultas necessárias, probabilidade de uso de medicamentos e demanda de serviços por região.

## ⚙️ Funcionalidades

### 1. Página Inicial
- Apresentação do projeto, seus objetivos e importância
- Visão geral das funcionalidades disponíveis

### 2. Treinamento de Modelos
- **Dados Sintéticos**: Opção para gerar e treinar modelos usando uma base de dados sintética
  - Permite definir o número de amostras para o treinamento (100-10000)
  - Visualização da distribuição dos dados gerados por tratamento e região
- **Carregar Dados Próprios**: Funcionalidade para treinar modelos com dados do usuário (em desenvolvimento)

### 3. Realizar Predições
- **Fazer Predições**: Interface para realizar previsões com base nos modelos treinados
  - Previsão do número de consultas médicas que um paciente realizará no ano
  - Estimativa da probabilidade de um paciente precisar de medicamentos além da terapia
  - Previsão da demanda de serviços por região
- **Visualizar Gráficos**: Análises visuais detalhadas dos dados
  - Distribuição de tipos de diagnóstico
  - Número de consultas por idade
  - Acesso a serviços por renda familiar
  - Evolução do número de diagnósticos ao longo dos anos
  - Distribuição de tratamentos por região
  - Filtros interativos por idade, ano e região

### 4. Análise de Dados
- Geração de dados realistas para simulação
- Pré-processamento de dados para análise
- Modelos de machine learning otimizados:
  - Regressão (número de consultas e demanda de serviços)
  - Classificação (necessidade de medicamentos)

## 🛠️ Tecnologias Utilizadas

- **Python**: Linguagem de programação principal
- **Streamlit**: Framework para criação de aplicações web interativas
- **pandas**: Manipulação e análise de dados
- **scikit-learn**: Bibliotecas para aprendizado de máquina
- **Plotly/Matplotlib/Seaborn**: Visualizações gráficas
- **NumPy**: Computação científica

## 💻 Instalação

Existem duas formas de instalar e executar o projeto:

### Usando script de inicialização

1. Clone o repositório:
   ```
   git clone https://github.com/seu-usuario/DataScience-AutismBrasil.git
   cd DataScience-AutismBrasil
   ```

2. Execute o script de inicialização:
   ```
   bash start.sh
   ```
   
   O script identificará automaticamente seu ambiente e instalará todas as dependências necessárias.

### Instalação manual

1. Clone o repositório:
   ```
   git clone https://github.com/seu-usuario/DataScience-AutismBrasil.git
   cd DataScience-AutismBrasil
   ```

2. Instale as dependências usando pip:
   ```
   pip install -r requirements.txt
   ```
   
   Ou usando pipenv (se disponível):
   ```
   pipenv install
   pipenv shell
   ```

3. Execute a aplicação:
   ```
   streamlit run app.py
   ```

## 🚀 Como Usar

1. **Inicie a aplicação** seguindo as instruções de instalação acima
2. Navegue até `http://localhost:8501` no seu navegador
3. **Treine os modelos** acessando a página "Treinamento de Modelos"
4. **Realize predições** na página "Realizar Predições"
5. **Explore visualizações** detalhadas com os filtros disponíveis

## 📁 Estrutura do Projeto

```
DataScience-AutismBrasil/
├── app.py                  # Aplicação principal Streamlit
├── requirements.txt        # Dependências do projeto
├── Pipfile                 # Configuração do pipenv
├── Pipfile.lock            # Lock de dependências do pipenv
├── start.sh                # Script de inicialização
├── modelos/                # Modelos treinados salvos
├── src/
│   ├── analises_visualizacoes.py    # Código para visualizações
│   ├── gerar_base_sintetica.py      # Geração de dados sintéticos
│   ├── ia/
│   │   ├── modelos.py               # Implementação dos modelos ML
│   │   └── pre-processamento.py     # Funções de pré-processamento
├── base_ajustada_realista_pre_processada.csv  # Dados de exemplo
```

## 👥 Desenvolvedores

- 🙋‍♂️ Everton Reis
- 🙋‍♀️ Bárbara Zamperete

---

## 📊 Exemplos de Análises

O sistema permite uma análise detalhada de vários aspectos do autismo no Brasil:

- Relação entre renda familiar e acesso a serviços de saúde
- Distribuição de diagnósticos por região e faixa etária
- Evolução temporal no número de diagnósticos
- Tipos de tratamento mais comuns por região
- Correlação entre idade e número de consultas

## 🔄 Fluxo de Trabalho Recomendado

1. Acesse a página inicial para entender o propósito do sistema
2. Vá para "Treinamento de Modelos" e gere dados sintéticos
3. Treine os modelos com parâmetros adequados
4. Utilize "Realizar Predições" para fazer análises
5. Explore as visualizações interativas e aplique filtros para insights específicos 
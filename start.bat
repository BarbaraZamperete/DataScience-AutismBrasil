@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo     Sistema de Analise de Autismo no Brasil
echo =====================================================
echo.
echo Este script ira configurar e iniciar o sistema automaticamente.
echo Desenvolvido por: Everton Reis e Barbara Zamperete
echo.
echo Iniciando configuracao...
echo.

REM Verificando Sistema Operacional
echo ==== Verificando Sistema Operacional ====
echo [DataScience AutismBrasil] Sistema operacional: Windows
echo.

REM Verificando Dependências
echo ==== Verificando Dependencias ====

REM Verificar se Python está instalado
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCESSO] Python encontrado
    set PYTHON_CMD=python
) else (
    python3 --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [SUCESSO] Python 3 encontrado
        set PYTHON_CMD=python3
    ) else (
        echo [ERRO] Python nao encontrado. Por favor, instale o Python 3 e tente novamente.
        pause
        exit /b 1
    )
)

REM Obter versão do Python
for /f "tokens=*" %%i in ('%PYTHON_CMD% --version 2^>^&1') do (
    set PYTHON_VERSION=%%i
)
echo [DataScience AutismBrasil] Versao do Python: %PYTHON_VERSION%

REM Verificar se Pipenv está instalado
pipenv --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCESSO] Pipenv encontrado. Usaremos ele para gerenciar as dependencias.
    set USE_PIPENV=true
) else (
    echo [AVISO] Pipenv nao encontrado. Usaremos pip para instalar as dependencias.
    set USE_PIPENV=false
)

REM Verificar se deve continuar automaticamente ou perguntar ao usuário
if "%USE_PIPENV%"=="false" (
    echo.
    set /p response=Pipenv nao encontrado. Deseja continuar com a instalacao usando pip? (S/n): 
    if /i "%response%"=="n" (
        echo [DataScience AutismBrasil] Instalacao cancelada pelo usuario.
        pause
        exit /b 0
    )
) else (
    echo [DataScience AutismBrasil] Pipenv encontrado. Continuando automaticamente com a instalacao...
)

REM Instalar dependências
echo.
echo ==== Instalando Dependencias ====

if "%USE_PIPENV%"=="true" (
    echo [DataScience AutismBrasil] Instalando dependencias com pipenv...
    
    if exist Pipfile (
        pipenv install
        if %errorlevel% neq 0 (
            echo [ERRO] Erro ao instalar dependencias com pipenv.
            pause
            exit /b 1
        )
        echo [SUCESSO] Dependencias instaladas com sucesso usando pipenv
    ) else (
        echo [ERRO] Arquivo Pipfile nao encontrado
        pause
        exit /b 1
    )
) else (
    echo [DataScience AutismBrasil] Instalando dependencias com pip...
    
    if exist requirements.txt (
        %PYTHON_CMD% -m pip install -r requirements.txt
        if %errorlevel% neq 0 (
            echo [ERRO] Erro ao instalar dependencias com pip.
            pause
            exit /b 1
        )
        echo [SUCESSO] Dependencias instaladas com sucesso usando pip
    ) else (
        echo [ERRO] Arquivo requirements.txt nao encontrado
        pause
        exit /b 1
    )
)

REM Verificar estrutura do projeto
echo.
echo ==== Verificando Estrutura do Projeto ====

REM Verificar se o arquivo principal existe
if not exist app.py (
    echo [ERRO] Arquivo app.py nao encontrado. Verifique se voce esta no diretorio correto.
    pause
    exit /b 1
)
    
REM Verificar se o diretório de modelos existe ou criar se necessário
if not exist modelos (
    echo [DataScience AutismBrasil] Criando diretorio 'modelos'...
    mkdir modelos
    echo [SUCESSO] Diretorio 'modelos' criado com sucesso
) else (
    echo [SUCESSO] Diretorio 'modelos' encontrado
)

REM Verificar se o diretório src existe
if not exist src (
    echo [ERRO] Diretorio 'src' nao encontrado. A estrutura do projeto parece estar incompleta.
    pause
    exit /b 1
) else (
    echo [SUCESSO] Diretorio 'src' encontrado
)

REM Decidir se deve abrir navegador automaticamente
if "%USE_PIPENV%"=="true" (
    echo [DataScience AutismBrasil] Iniciando aplicacao automaticamente...
    
    REM Abrir navegador em segundo plano
    start "" http://localhost:8501
    
    REM Iniciar a aplicação
    echo.
    echo ==== Iniciando a Aplicacao ====
    echo [DataScience AutismBrasil] Iniciando o servidor Streamlit...
    echo [DataScience AutismBrasil] Acesse a interface pelo navegador em: http://localhost:8501
    
    if "%USE_PIPENV%"=="true" (
        pipenv run streamlit run app.py
    ) else (
        %PYTHON_CMD% -m streamlit run app.py
    )
) else (
    echo.
    echo Configuracao concluida! A aplicacao sera iniciada em seguida.
    echo Apos iniciar, voce podera acessar a interface pelo navegador em: http://localhost:8501
    pause
    
    REM Iniciar a aplicação
    echo.
    echo ==== Iniciando a Aplicacao ====
    echo [DataScience AutismBrasil] Iniciando o servidor Streamlit...
    echo [DataScience AutismBrasil] Acesse a interface pelo navegador em: http://localhost:8501
    
    if "%USE_PIPENV%"=="true" (
        pipenv run streamlit run app.py
    ) else (
        %PYTHON_CMD% -m streamlit run app.py
    )
)

if %errorlevel% neq 0 (
    echo [ERRO] Erro ao iniciar a aplicacao.
    pause
    exit /b 1
)

endlocal 
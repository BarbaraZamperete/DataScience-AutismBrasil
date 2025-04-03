@echo off
setlocal enabledelayedexpansion

REM Script para iniciar o sistema de análise de autismo no Brasil
REM Desenvolvido por: Everton Reis e Bárbara Zamperete
REM Versão 1.1.0 - Atualizado em 03/04/2025

title Sistema de Analise de Autismo no Brasil

echo =====================================================
echo     Sistema de Analise de Autismo no Brasil
echo =====================================================
echo.
echo Este script ira configurar e iniciar o sistema automaticamente.
echo Desenvolvido por: Everton Reis e Barbara Zamperete
echo.
echo Iniciando configuracao...
echo.

REM Captura de erros - iniciando log
echo [%date% %time%] Iniciando script > debug_log.txt

REM Verificando Sistema Operacional
echo ==== Verificando Sistema Operacional ====
echo [DataScience AutismBrasil] Sistema operacional: Windows
echo [%date% %time%] Verificando sistema operacional Windows >> debug_log.txt
echo.

REM Verificando Dependências
echo ==== Verificando Dependencias ====
echo [%date% %time%] Verificando dependências >> debug_log.txt

REM Verificar se Python está instalado
echo Verificando Python...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCESSO] Python encontrado
    echo [%date% %time%] Python encontrado >> debug_log.txt
    set PYTHON_CMD=python
) else (
    echo [INFO] Python não encontrado, verificando Python3...
    echo [%date% %time%] Python não encontrado, verificando Python3 >> debug_log.txt
    python3 --version >nul 2>&1
    if %errorlevel% equ 0 (
        echo [SUCESSO] Python 3 encontrado
        echo [%date% %time%] Python3 encontrado >> debug_log.txt
        set PYTHON_CMD=python3
    ) else (
        echo [ERRO] Python nao encontrado. Por favor, instale o Python 3 e tente novamente.
        echo [%date% %time%] ERRO: Python não encontrado >> debug_log.txt
        echo.
        echo Pressione qualquer tecla para sair...
        pause > nul
        exit /b 1
    )
)

REM Obter versão do Python
for /f "tokens=*" %%i in ('%PYTHON_CMD% --version 2^>^&1') do (
    set PYTHON_VERSION=%%i
)
echo [DataScience AutismBrasil] Versao do Python: %PYTHON_VERSION%
echo [%date% %time%] %PYTHON_VERSION% >> debug_log.txt

REM Verificar se Pipenv está instalado
echo Verificando Pipenv...
%PYTHON_CMD% -m pipenv --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCESSO] Pipenv encontrado. Usaremos ele para gerenciar as dependencias.
    echo [%date% %time%] Pipenv encontrado >> debug_log.txt
    set USE_PIPENV=true
) else (
    echo [AVISO] Pipenv nao encontrado. Usaremos pip para instalar as dependencias.
    echo [%date% %time%] Pipenv não encontrado, usando pip >> debug_log.txt
    set USE_PIPENV=false
)

REM Verificar se deve continuar automaticamente ou perguntar ao usuário
if "%USE_PIPENV%"=="false" (
    echo.
    set /p response=Pipenv nao encontrado. Deseja continuar com a instalacao usando pip? (S/n): 
    if /i "%response%"=="n" (
        echo [DataScience AutismBrasil] Instalacao cancelada pelo usuario.
        echo [%date% %time%] Instalação cancelada pelo usuário >> debug_log.txt
        pause
        exit /b 0
    )
) else (
    echo [DataScience AutismBrasil] Pipenv encontrado. Continuando automaticamente com a instalacao...
)

REM Instalar dependências
echo.
echo ==== Instalando Dependencias ====
echo [%date% %time%] Instalando dependências >> debug_log.txt

if "%USE_PIPENV%"=="true" (
    echo [DataScience AutismBrasil] Instalando dependencias com pipenv...
    
    if exist Pipfile (
        %PYTHON_CMD% -m pipenv install
        if %errorlevel% neq 0 (
            echo [ERRO] Erro ao instalar dependencias com pipenv.
            echo [%date% %time%] ERRO: Falha na instalação com pipenv >> debug_log.txt
            echo.
            echo Detalhes do erro:
            %PYTHON_CMD% -m pipenv install
            echo.
            echo Pressione qualquer tecla para sair...
            pause > nul
            exit /b 1
        )
        echo [SUCESSO] Dependencias instaladas com sucesso usando pipenv
        echo [%date% %time%] Dependências instaladas com pipenv >> debug_log.txt
    ) else (
        echo [ERRO] Arquivo Pipfile nao encontrado
        echo [%date% %time%] ERRO: Pipfile não encontrado >> debug_log.txt
        echo.
        echo Pressione qualquer tecla para sair...
        pause > nul
        exit /b 1
    )
) else (
    echo [DataScience AutismBrasil] Instalando dependencias com pip...
    
    if exist requirements.txt (
        %PYTHON_CMD% -m pip install -r requirements.txt
        if %errorlevel% neq 0 (
            echo [ERRO] Erro ao instalar dependencias com pip.
            echo [%date% %time%] ERRO: Falha na instalação com pip >> debug_log.txt
            echo.
            echo Detalhes do erro:
            %PYTHON_CMD% -m pip install -r requirements.txt
            echo.
            echo Pressione qualquer tecla para sair...
            pause > nul
            exit /b 1
        )
        echo [SUCESSO] Dependencias instaladas com sucesso usando pip
        echo [%date% %time%] Dependências instaladas com pip >> debug_log.txt
    ) else (
        echo [ERRO] Arquivo requirements.txt nao encontrado
        echo [%date% %time%] ERRO: requirements.txt não encontrado >> debug_log.txt
        echo.
        echo Pressione qualquer tecla para sair...
        pause > nul
        exit /b 1
    )
)

REM Verificar estrutura do projeto
echo.
echo ==== Verificando Estrutura do Projeto ====
echo [%date% %time%] Verificando estrutura do projeto >> debug_log.txt

REM Verificar se o arquivo principal existe
if not exist app.py (
    echo [ERRO] Arquivo app.py nao encontrado. Verifique se voce esta no diretorio correto.
    echo [%date% %time%] ERRO: app.py não encontrado >> debug_log.txt
    echo.
    echo Pressione qualquer tecla para sair...
    pause > nul
    exit /b 1
)
    
REM Verificar se o diretório de modelos existe ou criar se necessário
if not exist modelos (
    echo [DataScience AutismBrasil] Criando diretorio 'modelos'...
    mkdir modelos
    echo [SUCESSO] Diretorio 'modelos' criado com sucesso
    echo [%date% %time%] Diretório 'modelos' criado >> debug_log.txt
) else (
    echo [SUCESSO] Diretorio 'modelos' encontrado
    echo [%date% %time%] Diretório 'modelos' encontrado >> debug_log.txt
)

REM Verificar se o diretório src existe
if not exist src (
    echo [ERRO] Diretorio 'src' nao encontrado. A estrutura do projeto parece estar incompleta.
    echo [%date% %time%] ERRO: diretório 'src' não encontrado >> debug_log.txt
    echo.
    echo Pressione qualquer tecla para sair...
    pause > nul
    exit /b 1
) else (
    echo [SUCESSO] Diretorio 'src' encontrado
    echo [%date% %time%] Diretório 'src' encontrado >> debug_log.txt
)

REM Verificar streamlit e tentar instalar se não estiver presente
echo Verificando se Streamlit está instalado...
%PYTHON_CMD% -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [AVISO] Streamlit não encontrado. Tentando instalar...
    echo [%date% %time%] Streamlit não encontrado, instalando >> debug_log.txt
    %PYTHON_CMD% -m pip install streamlit
    if %errorlevel% neq 0 (
        echo [ERRO] Falha ao instalar Streamlit.
        echo [%date% %time%] ERRO: Falha ao instalar Streamlit >> debug_log.txt
        echo.
        echo Pressione qualquer tecla para sair...
        pause > nul
        exit /b 1
    )
) else (
    echo [SUCESSO] Streamlit já está instalado
    echo [%date% %time%] Streamlit já instalado >> debug_log.txt
)

REM Decidir se deve abrir navegador automaticamente
if "%USE_PIPENV%"=="true" (
    echo [DataScience AutismBrasil] Iniciando aplicacao automaticamente...
    echo [%date% %time%] Iniciando aplicação automaticamente >> debug_log.txt
    
    REM Abrir navegador em segundo plano depois de um tempo
    echo Aguarde, o navegador será aberto em instantes...
    timeout /t 5 /nobreak > nul
    start "" http://localhost:8501
    
    REM Iniciar a aplicação
    echo.
    echo ==== Iniciando a Aplicacao ====
    echo [DataScience AutismBrasil] Iniciando o servidor Streamlit...
    echo [DataScience AutismBrasil] Acesse a interface pelo navegador em: http://localhost:8501
    echo [%date% %time%] Inicializando servidor Streamlit >> debug_log.txt
    
    if "%USE_PIPENV%"=="true" (
        %PYTHON_CMD% -m pipenv run streamlit run app.py
    ) else (
        %PYTHON_CMD% -m streamlit run app.py
    )
) else (
    echo.
    echo Configuracao concluida! A aplicacao sera iniciada em seguida.
    echo Apos iniciar, voce podera acessar a interface pelo navegador em: http://localhost:8501
    echo.
    echo Pressione qualquer tecla para continuar...
    pause > nul
    
    REM Iniciar a aplicação
    echo.
    echo ==== Iniciando a Aplicacao ====
    echo [DataScience AutismBrasil] Iniciando o servidor Streamlit...
    echo [DataScience AutismBrasil] Acesse a interface pelo navegador em: http://localhost:8501
    echo [%date% %time%] Inicializando servidor Streamlit >> debug_log.txt
    
    REM Abrir navegador em segundo plano
    timeout /t 5 /nobreak > nul
    start "" http://localhost:8501
    
    if "%USE_PIPENV%"=="true" (
        %PYTHON_CMD% -m pipenv run streamlit run app.py
    ) else (
        %PYTHON_CMD% -m streamlit run app.py
    )
)

if %errorlevel% neq 0 (
    echo [ERRO] Erro ao iniciar a aplicacao.
    echo [%date% %time%] ERRO: Falha ao iniciar aplicação >> debug_log.txt
    echo.
    echo Verifique o arquivo debug_log.txt para detalhes.
    echo.
    echo Pressione qualquer tecla para sair...
    pause > nul
    exit /b 1
)

echo [%date% %time%] Script finalizado com sucesso >> debug_log.txt
endlocal 
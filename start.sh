#!/usr/bin/env bash

#Script para iniciar o sistema de analise de autismo no Brasil criado por Everton Reis e Bárbara Zamperete
#Versão 1.0.0
#Data de criação: 04/04/2025
#Data de atualização: 04/04/2025
#leia o arquivo README.md para mais informações

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Função para exibir mensagens com formatação
print_message() {
    echo -e "${BLUE}[DataScience AutismBrasil]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCESSO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERRO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[AVISO]${NC} $1"
}

print_step() {
    echo -e "\n${PURPLE}==== $1 ====${NC}"
}

# Verificar o sistema operacional
check_os() {
    print_step "Verificando Sistema Operacional"
    
    case "$(uname -s)" in
        Linux*)     
            print_message "Sistema operacional: Linux"
            OS="linux"
            ;;
        Darwin*)    
            print_message "Sistema operacional: macOS"
            OS="macos"
            ;;
        CYGWIN*|MINGW*|MSYS*) 
            print_message "Sistema operacional: Windows"
            OS="windows"
            ;;
        *)          
            print_message "Sistema operacional: Outro"
            OS="outro"
            ;;
    esac

    # Verificação adicional para Windows (PowerShell)
    if [ "$OS" = "outro" ] && command -v powershell.exe &> /dev/null; then
        print_message "Detectado PowerShell no Windows"
        OS="windows"
    fi
}

# Verificar e instalar dependências
check_dependencies() {
    print_step "Verificando Dependências"

    # Verificar se Python está instalado
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        print_success "Python 3 encontrado"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        print_success "Python encontrado"
    else
        print_error "Python não encontrado. Por favor, instale o Python 3 e tente novamente."
        exit 1
    fi

    # Obter versão do Python
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    print_message "Versão do Python: $PYTHON_VERSION"

    # Verificar se Pipenv está instalado
    if command -v pipenv &> /dev/null; then
        print_success "Pipenv encontrado. Usaremos ele para gerenciar as dependências."
        USE_PIPENV=true
    else
        print_warning "Pipenv não encontrado. Usaremos pip para instalar as dependências."
        USE_PIPENV=false
    fi
}

# Instalar dependências com pipenv ou pip
install_dependencies() {
    print_step "Instalando Dependências"

    if [ "$USE_PIPENV" = true ]; then
        print_message "Instalando dependências com pipenv..."
        
        if [ -f "Pipfile" ]; then
            pipenv install
            if [ $? -ne 0 ]; then
                print_error "Erro ao instalar dependências com pipenv."
                exit 1
            fi
            print_success "Dependências instaladas com sucesso usando pipenv"
        else
            print_error "Arquivo Pipfile não encontrado"
            exit 1
        fi
    else
        print_message "Instalando dependências com pip..."
        
        if [ -f "requirements.txt" ]; then
            $PYTHON_CMD -m pip install -r requirements.txt
            if [ $? -ne 0 ]; then
                print_error "Erro ao instalar dependências com pip."
                exit 1
            fi
            print_success "Dependências instaladas com sucesso usando pip"
        else
            print_error "Arquivo requirements.txt não encontrado"
            exit 1
        fi
    fi
}

# Verificar se diretórios e arquivos necessários existem
check_project_structure() {
    print_step "Verificando Estrutura do Projeto"
    
    # Verificar se o arquivo principal existe
    if [ ! -f "app.py" ]; then
        print_error "Arquivo app.py não encontrado. Verifique se você está no diretório correto."
        exit 1
    fi
    
    # Verificar se o diretório de modelos existe ou criar se necessário
    if [ ! -d "modelos" ]; then
        print_message "Criando diretório 'modelos'..."
        mkdir -p modelos
        print_success "Diretório 'modelos' criado com sucesso"
    else
        print_success "Diretório 'modelos' encontrado"
    fi

    # Verificar se o diretório src existe
    if [ ! -d "src" ]; then
        print_error "Diretório 'src' não encontrado. A estrutura do projeto parece estar incompleta."
        exit 1
    else
        print_success "Diretório 'src' encontrado"
    fi
}

# Iniciar a aplicação
start_application() {
    print_step "Iniciando a Aplicação"
    
    print_message "Iniciando o servidor Streamlit..."
    print_message "Acesse a interface pelo navegador em: http://localhost:8501"
    
    if [ "$USE_PIPENV" = true ]; then
        pipenv run streamlit run app.py
    else
        $PYTHON_CMD -m streamlit run app.py
    fi
    
    if [ $? -ne 0 ]; then
        print_error "Erro ao iniciar a aplicação."
        exit 1
    fi
}

# Abrir o navegador (opcional)
open_browser() {
    # Espera 3 segundos para o servidor iniciar
    sleep 3
    
    print_message "Tentando abrir o navegador automaticamente..."
    
    if [ "$OS" = "linux" ]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8501 &
        elif command -v sensible-browser &> /dev/null; then
            sensible-browser http://localhost:8501 &
        fi
    elif [ "$OS" = "macos" ]; then
        open http://localhost:8501 &
    elif [ "$OS" = "windows" ]; then
        if command -v start &> /dev/null; then
            start http://localhost:8501 &
        elif command -v cmd.exe &> /dev/null; then
            cmd.exe /c start http://localhost:8501 &
        fi
    fi
}

# Mostrar informações iniciais
show_welcome() {
    clear
    echo -e "${CYAN}"
    echo "====================================================="
    echo "    Sistema de Análise de Autismo no Brasil"
    echo "====================================================="
    echo -e "${NC}"
    echo "Este script irá configurar e iniciar o sistema automaticamente."
    echo "Desenvolvido por: Everton Reis e Bárbara Zamperete"
    echo -e "\nIniciando configuração...\n"
    sleep 1
}

# Função principal
main() {
    show_welcome
    check_os
    check_dependencies
    
    # Se o pipenv estiver instalado, continuar automaticamente
    # Se não, perguntar ao usuário
    if [ "$USE_PIPENV" = false ]; then
        echo -e "\n${YELLOW}Pipenv não encontrado. Deseja continuar com a instalação usando pip? (S/n)${NC}"
        read -r response
        if [[ "$response" =~ ^[Nn]$ ]]; then
            print_message "Instalação cancelada pelo usuário."
            exit 0
        fi
    else
        print_message "Pipenv encontrado. Continuando automaticamente com a instalação..."
    fi
    
    install_dependencies
    check_project_structure
    
    # Se estiver usando pipenv, iniciar diretamente
    # Se não, perguntar ao usuário
    if [ "$USE_PIPENV" = true ]; then
        print_message "Iniciando aplicação automaticamente..."
        # Tentar abrir o navegador em segundo plano
        open_browser &
        start_application
    else
        echo -e "\n${YELLOW}Configuração concluída! A aplicação será iniciada em seguida."
        echo -e "Após iniciar, você poderá acessar a interface pelo navegador em: http://localhost:8501"
        echo -e "Pressione Enter para continuar...${NC}"
        read -r
        
        start_application
    fi
}

# Executar a função principal
main 
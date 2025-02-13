import streamlit as st
import pandas as pd
import numpy as np


def main(n_samples):
    df = gerar_base_sintetica(n_samples)
    return df



def gerar_base_sintetica(n_samples):
    np.random.seed(42)
    diagnosticos = ['Leve', 'Moderado', 'Grave']
    tratamentos = ['Terapias', 'Medicamentos', 'Ambos']
    regioes = ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul']
    apoio_familiar = ['Sim', 'Não']
    renda_familiar = ['Baixa', 'Média', 'Alta']
    tipos_servicos = ['Público', 'Privado']
    zona = ['Urbana', 'Rural']

    # Geração de Idade
    idades = np.random.randint(5, 45, size=n_samples)

    # Geração de Renda Familiar
    renda = np.random.choice(renda_familiar, size=n_samples, p=[0.5, 0.4, 0.1])

    # Geração de Ano de Diagnóstico entre 2012 e 2024 com aumento no número de diagnósticos
    base_probs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  # Tendência crescente
    base_probs = base_probs / base_probs.sum()  # Normaliza para que a soma seja 1
    anos = np.random.choice(range(2012, 2025), size=n_samples, p=base_probs)  # Aumenta a probabilidade de anos mais recentes

    # Geração de Número de Consultas com base na Renda e Ano de Diagnóstico
    numero_consultas = []
    for ano in anos:
        # Calcular o tempo desde o diagnóstico (2024 - ano)
        tempo_diagnostico = 2024 - ano
        
        # Aumentar o número de consultas com base na renda e no tempo desde o diagnóstico
        if renda[np.random.randint(0, n_samples)] == 'Alta':
            consultas = np.random.randint(1, 15) + tempo_diagnostico  # Renda alta permite mais consultas
        else:
            consultas = np.random.randint(1, 10) + tempo_diagnostico  # Renda média ou baixa
        
        numero_consultas.append(consultas)

    # Geração de Tipo de Diagnóstico com base na Idade
    tipo_diagnostico = []
    for idade in idades:
        if idade < 15:
            tipo_diagnostico.append(np.random.choice(diagnosticos, p=[0.2, 0.3, 0.5]))  # Mais grave em crianças
        elif idade < 30:
            tipo_diagnostico.append(np.random.choice(diagnosticos, p=[0.4, 0.5, 0.1]))  # Moderado
        else:
            tipo_diagnostico.append(np.random.choice(diagnosticos, p=[0.5, 0.4, 0.1]))  # Mais leve em adultos

    # Geração de Tratamento com base no Tipo de Diagnóstico e Renda
    tratamento = []
    for diagnostico, r in zip(tipo_diagnostico, renda):
        if diagnostico == 'Grave':
            tratamento.append(np.random.choice(tratamentos, p=[0.3, 0.3, 0.4]))  # Mais medicamentos e ambos
        elif diagnostico == 'Moderado':
            tratamento.append(np.random.choice(tratamentos, p=[0.1, 0.5, 0.4]))  
        else:
            tratamento.append(np.random.choice(tratamentos, p=[0.6, 0.3, 0.1]))  # Mais terapias

    # Geração de Acesso a Serviços com base na Renda Familiar
    acesso_servicos = []
    for r in renda:
        if r == 'Alta':
            acesso_servicos.append(np.random.choice(['Sim', 'Não'], p=[0.9, 0.1]))
        elif r == 'Média':
            acesso_servicos.append(np.random.choice(['Sim', 'Não'], p=[0.7, 0.3]))
        else:  # Renda Baixa
            acesso_servicos.append(np.random.choice(['Sim', 'Não'], p=[0.1, 0.9]))

    # Geração de Tipo de Servico com base na Renda Familiar
    tipos_servicos = []
    for r in renda:
        if r == 'Alta':
            tipos_servicos.append(np.random.choice(['Público', 'Privado'], p=[0.9, 0.1]))
        elif r == 'Média':
            tipos_servicos.append(np.random.choice(['Público', 'Privado'], p=[0.7, 0.3]))
        else:  # Renda Baixa
            tipos_servicos.append(np.random.choice(['Público', 'Privado'], p=[0.1, 0.9]))

    # Geração de Apoio Familiar com base na Renda Familiar
    apoio_familiar = []
    for r in renda:
        if r == 'Alta':
            apoio_familiar.append(np.random.choice(['Sim', 'Não'], p=[0.9, 0.1]))
        elif r == 'Média':
            apoio_familiar.append(np.random.choice(['Sim', 'Não'], p=[0.7, 0.3]))
        else:  # Renda Baixa
            apoio_familiar.append(np.random.choice(['Sim', 'Não'], p=[0.1, 0.9]))

    zona = []
    for r in renda:
        if r == 'Alta':
            zona.append(np.random.choice(['Urbana', 'Rural'], p=[0.9, 0.1]))
        elif r == 'Média':
            zona.append(np.random.choice(['Urbana', 'Rural'], p=[0.7, 0.3]))
        else:  # Renda Baixa
            zona.append(np.random.choice(['Urbana', 'Rural'], p=[0.1, 0.9]))


    data_ajustada = {
        'ID': range(1, n_samples + 1),
        'Idade': idades,
        'Tipo_de_Diagnostico': tipo_diagnostico,
        'Ano_Diagnostico': anos,
        'Tratamento': tratamento,
        'Acesso_a_Servicos': acesso_servicos,
        'Regiao': np.random.choice(regioes, size=n_samples, p=[0.1, 0.3, 0.1, 0.4, 0.1]),
        'Apoio_Familiar': apoio_familiar,
        'Renda_Familiar': renda,
        'Tipo_de_Servico': tipos_servicos,
        'Zona': zona,
        'Numero_de_Consultas': numero_consultas
    }

    df = pd.DataFrame(data_ajustada)
    return df


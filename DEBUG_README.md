# Configurações de Debug do VS Code

## Estrutura do Projeto

```
lwe/
├── .vscode/
│   ├── launch.json       # Configurações de debug
│   ├── settings.json     # Configurações do workspace
│   └── tasks.json        # Tasks do VS Code
├── venv/                 # Virtual environment unificado
├── requirements.txt      # Dependências unificadas
├── ckks/
│   ├── main.py          # Implementação CKKS
│   └── test_main.py     # Testes CKKS
└── tfhe/
    ├── main.py          # Implementação TFHE
    └── test_main.py     # Testes TFHE
```

## Configurações de Debug Disponíveis

#### Para CKKS:
- **Debug CKKS main.py**: Executa e debugga o arquivo principal do CKKS
- **Debug CKKS Tests**: Executa todos os testes do CKKS com debug

#### Para TFHE:
- **Debug TFHE main.py**: Executa e debugga o arquivo principal do TFHE
- **Debug TFHE Tests**: Executa todos os testes do TFHE com debug

#### Geral:
- **Debug Current Python File**: Debugga o arquivo Python atualmente aberto

## Como Usar

### Iniciar Debug
1. **Via Debug Panel (Ctrl+Shift+D)**:
   - Selecione a configuração no dropdown
   - Clique no botão play verde

2. **Via Atalho**: Pressione F5

### Adicionar Breakpoints
- Clique na margem esquerda do editor (ao lado dos números de linha)
- Ou pressione F9 na linha desejada

### Controles Durante Debug
- **F10**: Próxima linha
- **F11**: Entrar em função
- **F5**: Continuar
- **Shift+F5**: Parar

## Tasks Disponíveis

Acesse via **Ctrl+Shift+P > Tasks: Run Task**:

- **Run CKKS Tests**: Executa testes do CKKS
- **Run CKKS main.py**: Executa o programa principal do CKKS
- **Run TFHE Tests**: Executa testes do TFHE
- **Run TFHE main.py**: Executa o programa principal do TFHE

## Comandos Úteis

```bash
# Ativar ambiente CKKS
cd ckks && source venv/bin/activate

# Ativar ambiente TFHE  
cd tfhe && source venv/bin/activate

# Executar testes manualmente
pytest test_main.py -v
```

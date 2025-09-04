#!/bin/bash
# Script para ativar o virtual environment unificado (pasta CKKS)
# Execute com: source activate_env.sh

echo "Ativando virtual environment unificado do projeto LWE..."
cd ..
source venv/bin/activate
cd ckks
echo "Virtual environment ativado! (Você está na pasta ckks)"
echo "Python: $(which python)"
echo ""
echo "Para executar os testes CKKS:"
echo "  pytest test_main.py -v"
echo ""
echo "Para executar o programa principal:"
echo "  python main.py"

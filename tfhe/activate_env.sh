#!/bin/bash
# Script para ativar o virtual environment da pasta tfhe
# Execute com: source activate_env.sh

echo "Ativando virtual environment do TFHE..."
source venv/bin/activate
echo "Virtual environment ativado!"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
echo ""
echo "Para executar os testes, use:"
echo "  pytest test_main.py -v"
echo "  pytest test_main.py -v --cov=main"
echo ""
echo "Para executar o programa principal:"
echo "  python main.py"

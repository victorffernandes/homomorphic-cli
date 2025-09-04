#!/bin/bash
# Script para ativar o virtual environment unificado do projeto LWE
# Execute com: source activate_env.sh

echo "Ativando virtual environment unificado do projeto LWE..."
source venv/bin/activate
echo "Virtual environment ativado!"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
echo ""
echo "Para executar os testes CKKS:"
echo "  cd ckks && pytest test_main.py -v"
echo ""
echo "Para executar os testes TFHE:"
echo "  cd tfhe && pytest test_main.py -v"
echo ""
echo "Para executar os programas principais:"
echo "  cd ckks && python main.py"
echo "  cd tfhe && python main.py"

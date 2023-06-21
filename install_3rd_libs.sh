# install 3rd libs
#!/bin/bash
# Prompt user to enter library name
echo "Which library would you like to install? (spqr/awq)"
read LIBRARY
ROOT=$PWD
cd lptorch/config && python3 add_extra_q.py $LIBRARY
cd $ROOT
# Install library based on user input
if [[ $LIBRARY == "spqr" ]]; then
    echo "SPQR Not integrated yet"
#   cd 3rd_party/SpQR
#   pip install -r requirements.txt
#   pip install safetensors==0.3.0 datasets==2.10.1 sentencepiece==0.1.97
elif [[ $LIBRARY == "awq" ]]; then
    echo "AWQ Not integrated yet"
    # cd 3rd_party/llm-awq/
    # pip install -e .
    # cd awq/kernels
    # pip install .
else
  echo "Invalid library name. Please enter either 'spqr' or 'awq'."
fi
cd $ROOT
# reinstall lptorch
pip install .
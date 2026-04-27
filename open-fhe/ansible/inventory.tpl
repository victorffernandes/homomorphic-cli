[fhe]
${public_ip} ansible_user=ubuntu ansible_ssh_private_key_file=${ssh_private_key_path}

[fhe:vars]
ansible_python_interpreter=/usr/bin/python3
ansible_ssh_common_args=-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null

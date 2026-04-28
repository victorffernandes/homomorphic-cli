# LSSVM Cloud Playbook — Common Commands

All commands assume:
- `cd /home/main/Documents/Projects/lwe/open-fhe/ansible`
- Inventory at `inventory.ini` (rendered by Terraform)
- SSH key at `~/.ssh/oci-keys-rsa`
- Repo root: `/home/main/Documents/Projects/lwe`

Set once per shell:
```
export REPO_ROOT=/home/main/Documents/Projects/lwe
export INSTANCE_IP=$(awk 'NR==2 {print $1}' inventory.ini)
```

---

## Full provision (first time, or after `terraform apply`)

```
ANSIBLE_HOST_KEY_CHECKING=False \
ansible-playbook -i inventory.ini site.yml \
  --extra-vars "repo_root=$REPO_ROOT"
```

Builds OpenFHE C++, builds openfhe-python, syncs code, installs Python deps. ~15–25 min on cold instance.

---

## Redeploy code only (fast — seconds)

Pushes `open-fhe/*.py` + `requirements.txt` updates without rebuilding OpenFHE.

```
ansible-playbook -i inventory.ini site.yml --tags sync \
  --extra-vars "repo_root=$REPO_ROOT"
```

Or raw rsync (skips Ansible overhead entirely):

```
rsync -av --delete \
  --exclude __pycache__ --exclude '*.pyc' \
  --exclude terraform --exclude ansible \
  -e "ssh -i ~/.ssh/oci-keys-rsa" \
  $REPO_ROOT/open-fhe/ \
  ubuntu@$INSTANCE_IP:/opt/lssvm/app/
```

---

## Run scripts on the VM

`run.sh` is a generic dispatcher. Default script `lssvm_cipher.py`; pass any `*.py` as first arg to switch.

```
# Default: single-client cipher LSSVM
ssh -i ~/.ssh/oci-keys-rsa ubuntu@$INSTANCE_IP /opt/lssvm/app/run.sh

# Federated training, k=3 clients
ssh -i ~/.ssh/oci-keys-rsa ubuntu@$INSTANCE_IP /opt/lssvm/app/run.sh federated_lssvm.py 3

# Federated inference, k=3
ssh -i ~/.ssh/oci-keys-rsa ubuntu@$INSTANCE_IP /opt/lssvm/app/run.sh federated_infer.py 3
```

`run.sh` exports `OMP_NUM_THREADS=$(nproc)` by default + sets `OMP_PROC_BIND=spread` `OMP_PLACES=cores` so OpenFHE uses all cores. Pins `OPENBLAS/MKL=1` to avoid BLAS-vs-OpenMP contention.

### Tune thread count

Three knobs (highest precedence first):

```
# 1. Python CLI flag (per-invocation, strips itself from argv)
./run.sh federated_lssvm.py --threads=2 3

# 2. FHE_THREADS env (read by parallel.py)
FHE_THREADS=2 ./run.sh federated_lssvm.py 3

# 3. OMP_NUM_THREADS env (also honored)
OMP_NUM_THREADS=2 ./run.sh federated_lssvm.py 3
```

### Verify cores saturate

In one ssh session run training, in another:
```
ssh -i ~/.ssh/oci-keys-rsa ubuntu@$INSTANCE_IP htop
```
Expect all `$(nproc)` cores near 100% during QR / back-sub phases.

### Stream logs to a file

```
ssh -i ~/.ssh/oci-keys-rsa ubuntu@$INSTANCE_IP \
  '/opt/lssvm/app/run.sh federated_lssvm.py 3 2>&1 | tee -a ~/lssvm.log'
```

---

## Bump OpenFHE version

Edit `site.yml` `openfhe_version:` (and/or pass `--extra-vars openfhe_version=vX.Y.Z`). Then nuke build + install dirs and re-run:

```
ssh -i ~/.ssh/oci-keys-rsa ubuntu@$INSTANCE_IP \
  'sudo rm -rf /opt/build/openfhe /opt/build/openfhe-python/build /usr/local/lib/OpenFHE /usr/local/lib/libOPENFHE* /usr/local/include/openfhe'

ansible-playbook -i inventory.ini site.yml \
  --extra-vars "repo_root=$REPO_ROOT"
```

openfhe-python `main` must match the C++ version. Pin both if needed via `openfhe_python_version` extra-var.

---

## Rebuild only Python bindings (after pulling new openfhe-python)

```
ssh -i ~/.ssh/oci-keys-rsa ubuntu@$INSTANCE_IP \
  'sudo rm -rf /opt/build/openfhe-python/build'
ansible-playbook -i inventory.ini site.yml \
  --extra-vars "repo_root=$REPO_ROOT"
```

Idempotent steps before the build are skipped automatically.

---

## Debug failing run

```
ssh -i ~/.ssh/oci-keys-rsa ubuntu@$INSTANCE_IP
sudo cloud-init status --long
sudo tail -100 /var/log/cloud-init-output.log

# inside venv
source /opt/lssvm/venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/lib
python -c 'import openfhe; print(openfhe.__file__)'
```

---

## Tear down

```
cd /home/main/Documents/Projects/lwe/open-fhe/terraform
terraform destroy
```

Removes VM, VCN, subnet, IGW. `terraform.tfstate` retained locally.

---

## Tag reference

- `sync` / `code` → push source + requirements.txt only

Other tasks have no tags (always run on full play).

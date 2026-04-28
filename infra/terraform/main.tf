terraform {
  required_providers {
    oci   = { source = "oracle/oci", version = "~> 6.0" }
    local = { source = "hashicorp/local", version = "~> 2.5" }
    null  = { source = "hashicorp/null", version = "~> 3.2" }
  }
}

provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

# Resolve latest Ubuntu 22.04 ARM64 platform image dynamically
data "oci_core_images" "ubuntu_arm" {
  compartment_id           = var.compartment_id
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "22.04"
  shape                    = "VM.Standard.A1.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

locals {
  ubuntu_image_id = data.oci_core_images.ubuntu_arm.images[0].id
  ansible_dir     = "${path.module}/../ansible"
  inventory_path  = "${local.ansible_dir}/inventory.ini"
}

module "infrastructure" {
  source = "./resources"

  compartment_id      = var.compartment_id
  tenancy_ocid        = var.tenancy_ocid
  ubuntu_image_id     = local.ubuntu_image_id
  ssh_public_key      = var.ssh_public_key
  instance_name       = var.instance_name
  instance_ocpus      = var.instance_ocpus
  instance_memory_gb  = var.instance_memory_gb
  boot_volume_size_gb = var.boot_volume_size_gb
}

resource "local_file" "ansible_inventory" {
  filename = local.inventory_path
  content = templatefile("${local.ansible_dir}/inventory.tpl", {
    public_ip            = module.infrastructure.public_ip
    ssh_private_key_path = var.ssh_private_key_path
  })
  file_permission = "0644"
}

resource "null_resource" "ansible_provision" {
  triggers = {
    instance_id = module.infrastructure.instance_ocid
    playbook    = filemd5("${local.ansible_dir}/site.yml")
  }

  provisioner "local-exec" {
    working_dir = local.ansible_dir
    command     = <<-EOT
      # Wait for SSH to accept connections. cloud-init may report "error"
      # because of transient DNS at first boot (packages: list fails) — that's OK,
      # Ansible re-runs apt with retries and is the source of truth for setup.
      for i in $(seq 1 60); do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
               -i ${var.ssh_private_key_path} \
               ubuntu@${module.infrastructure.public_ip} 'true' 2>/dev/null; then
          echo "ssh ready"
          break
        fi
        echo "waiting for ssh... ($i/60)"
        sleep 5
      done
      ansible-galaxy collection install -r requirements.yml
      ANSIBLE_HOST_KEY_CHECKING=False \
      ansible-playbook -i ${local.inventory_path} site.yml \
        --extra-vars "openfhe_version=${var.openfhe_version} repo_root=${abspath("${path.module}/../..")}"
    EOT
  }

  depends_on = [local_file.ansible_inventory]
}

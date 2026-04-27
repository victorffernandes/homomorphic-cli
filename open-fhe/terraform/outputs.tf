output "public_ip" {
  description = "Public IP address of the instance"
  value       = module.infrastructure.public_ip
}

output "instance_ocid" {
  description = "OCID of the compute instance"
  value       = module.infrastructure.instance_ocid
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh ubuntu@${module.infrastructure.public_ip}"
}

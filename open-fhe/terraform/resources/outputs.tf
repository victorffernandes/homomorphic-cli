output "public_ip" {
  description = "Public IP address of the inference instance"
  value       = oci_core_instance.main.public_ip
}

output "instance_ocid" {
  description = "OCID of the compute instance"
  value       = oci_core_instance.main.id
}

terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
  subscription_id = var.subscription_id
}

resource "azurerm_resource_group" "nightshift" {
  name     = "nightshift-rg"
  location = var.location
  tags     = local.tags
}

locals {
  tags = {
    project = "nightshift"
    purpose = "arm-ai-optimization-challenge-2026"
  }
}

resource "azurerm_virtual_network" "vnet" {
  name                = "nightshift-vnet"
  address_space       = ["10.42.0.0/16"]
  location            = azurerm_resource_group.nightshift.location
  resource_group_name = azurerm_resource_group.nightshift.name
  tags                = local.tags
}

resource "azurerm_subnet" "subnet" {
  name                 = "nightshift-subnet"
  resource_group_name  = azurerm_resource_group.nightshift.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.42.1.0/24"]
}

resource "azurerm_network_security_group" "nsg" {
  name                = "nightshift-nsg"
  location            = azurerm_resource_group.nightshift.location
  resource_group_name = azurerm_resource_group.nightshift.name
  tags                = local.tags

  security_rule {
    name                       = "ssh"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = var.allowed_ip
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "llama-server"
    priority                   = 110
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8080"
    source_address_prefix      = var.allowed_ip
    destination_address_prefix = "*"
  }
}

resource "azurerm_public_ip" "pip" {
  name                = "nightshift-pip"
  location            = azurerm_resource_group.nightshift.location
  resource_group_name = azurerm_resource_group.nightshift.name
  allocation_method   = "Static"
  sku                 = "Standard"
  zones               = [var.zone]
  tags                = local.tags
}

resource "azurerm_network_interface" "nic" {
  name                = "nightshift-nic"
  location            = azurerm_resource_group.nightshift.location
  resource_group_name = azurerm_resource_group.nightshift.name
  tags                = local.tags

  ip_configuration {
    name                          = "primary"
    subnet_id                     = azurerm_subnet.subnet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.pip.id
  }
}

resource "azurerm_network_interface_security_group_association" "nsg_assoc" {
  network_interface_id      = azurerm_network_interface.nic.id
  network_security_group_id = azurerm_network_security_group.nsg.id
}

# The model disk: survives VM respins/resizes. Download once, snapshot, reuse.
resource "azurerm_managed_disk" "models" {
  name                 = "nightshift-models"
  location             = azurerm_resource_group.nightshift.location
  resource_group_name  = azurerm_resource_group.nightshift.name
  storage_account_type = "PremiumV2_LRS"
  create_option        = "Empty"
  disk_size_gb         = var.model_disk_gb
  disk_iops_read_write = 10000
  disk_mbps_read_write = 800
  zone                 = var.zone
  tags                 = local.tags
}

resource "azurerm_linux_virtual_machine" "vm" {
  name                = "nightshift-vm"
  location            = azurerm_resource_group.nightshift.location
  resource_group_name = azurerm_resource_group.nightshift.name
  size                = var.vm_size
  zone                = var.zone
  admin_username      = var.admin_username
  network_interface_ids = [azurerm_network_interface.nic.id]
  tags                = local.tags

  # Spot support: flips on when quota allows (sponsorship subs often cap spot).
  priority        = var.use_spot ? "Spot" : "Regular"
  eviction_policy = var.use_spot ? "Deallocate" : null
  max_bid_price   = var.use_spot ? -1 : null

  admin_ssh_key {
    username   = var.admin_username
    public_key = file(var.ssh_public_key_path)
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "StandardSSD_LRS"
    disk_size_gb         = 128
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "ubuntu-24_04-lts"
    sku       = "server-arm64"
    version   = "latest"
  }

  custom_data = base64encode(file("${path.module}/cloud-init.yaml"))
}

resource "azurerm_virtual_machine_data_disk_attachment" "models" {
  managed_disk_id    = azurerm_managed_disk.models.id
  virtual_machine_id = azurerm_linux_virtual_machine.vm.id
  lun                = 0
  caching            = "None" # PremiumV2 requires None; llama.cpp mmap does its own caching
}

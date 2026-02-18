###
#Filename -  Authenticating to Azure ML.py
###

'''
Options for authenticating in Azure Machine Learning
This reading will guide you through the following methods:

Interactive login authentication
Azure CLI authentication
Managed service identity (MSI) authentication
Service principal authentication
Token authentication
'''

#Step-by-step authentication instructions

##############################################
#Step 1: Open the authentication notebook
##############################################
#Access the notebook:
'''
Go to https://azure.com  and sign in if prompted.
Open your Azure ML workspace.
Navigate to Notebooks and switch to the Samples tab.
Go to SDK v1 > how-to-use-azureml > manage-azureml-service > authentication-in-azureml.
Clone the authentication-in-azureml.ipynb notebook and its dependencies into your workspace.
'''

#Prepare the environment:
#Set the kernel to Python 3.8-AzureML.

##########################################
#Step 2: Use an authentication method
##########################################
#Method 1: Interactive login authentication
#Interactive login is the default method and ideal for quick access during development.
#1. Import the required module:
from azureml.core import Workspace

#2. Authenticate:
'''
Interactive authentication is the default mode when using Azure ML SDK.
When you connect to your workspace using workspace.from_config, you will get an interactive 
login dialog.

Also, if you explicitly specify the subscription ID, resource group and workspace name, 
you will get the dialog.
ws = Workspace(subscription_id="my-subscription-id",
               resource_group="my-ml-rg",
               workspace_name="my-ml-workspace")
'''
ws = Workspace.from_config()

#Method 2 : Azure CLI authentication
from azureml.core.authentication import AzureCliAuthentication

cli_auth = AzureCliAuthentication()

ws = Workspace(subscription_id="my-subscription-id",
               resource_group="my-ml-rg",
               workspace_name="my-ml-workspace",
               auth=cli_auth)

print("Found workspace {} at location {}".format(ws.name, ws.location))

#Method 3: Managed service identity (MSI) authentication
from azureml.core.authentication import MsiAuthentication

msi_auth = MsiAuthentication()

ws = Workspace(subscription_id="my-subscription-id",
               resource_group="my-ml-rg",
               workspace_name="my-ml-workspace",
               auth=msi_auth)

print("Found workspace {} at location {}".format(ws.name, ws.location))




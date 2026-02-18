###
#Filename -  Deploying wirh Azure Kubernetes Service.py
###

'''
Note: you will need 12 virtual CPUs to do this activity. The free Azure account only has 10. 
If you don’t have enough virtual CPUs to complete this activity, feel free to skip it.

By the end of this activity, you will be able to:
Set up an Azure Machine Learning workspace.
Deploy models to Azure Kubernetes Service.
Test deployed models for reliability and performance.

Step-by-step guide to deploy models with Azure Kubernetes Service
This reading will guide you through the following steps:

Step 1: Access your workspace.
Step 2: Open the deployment notebook.
Step 3: Import and initialize.
Step 4: Register a model.
Step 5: Define the deployment environment.
Step 6: Write the entry script.
Step 7: Create inference configuration.
Step 8: Provision the Azure Kubernetes Service cluster.
Step 9: Deploy the model.
Step 10: Clean up resources.
'''
#######################################
#Step 1: Access your workspace
#######################################
#Go to https://ml.azure.com and sign in if prompted.
#Enter your Azure Machine Learning workspace.

#######################################
#Step 2: Open the deployment notebook
#######################################
'''
Navigate to Notebooks under the authoring section.
Switch to the Samples tab in the File Explorer.
Go to SDK v1 > how-to-use-azureml > deployment > production-deploy-to-aks.
'''
#######################################
#Step 3: Import and initialize
#######################################
from azureml.core import Workspace
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.model import Model

import azureml.core
print(azureml.core.VERSION)

#Get workspace
#Load existing workspace from the config file info.
from azureml.core.workspace import Workspace

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

#######################################
#Step 4: Register the model
#######################################
#Register an existing trained model, add descirption and tags.
#Register the model
from azureml.core.model import Model
model = Model.register(model_path = "sklearn_regression_model.pkl", # this points to a local file
                       model_name = "sklearn_regression_model.pkl", # this is the name the model is registered as
                       tags = {'area': "diabetes", 'type': "regression"},
                       description = "Ridge regression model to predict diabetes",
                       workspace = ws)

print(model.name, model.description, model.version)

#######################################
#Step 5: Define the deployment environment
#######################################
'''
Create a custom environment:
Specify Conda dependencies such as “numpy,” “scikit-learn,” and “scipy.”
Include Pip dependencies such as “Azure ML Defaults” and “Inference Schema.”
Optionally, use a custom docker image for further customization.
'''
#Create the Environment
#Create an environment that the model will be deployed with
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies 

conda_deps = CondaDependencies.create(conda_packages=['numpy','scikit-learn==0.22.1','scipy'], pip_packages=['azureml-defaults', 'inference-schema'])
myenv = Environment(name='myenv')
myenv.python.conda_dependencies = conda_deps

#Use a custom Docker image
'''
You can also specify a custom Docker image to be used as base image if you don't want to use the 
default base image provided by Azure ML. Please make sure the custom Docker image has 
Ubuntu >= 16.04, Conda >= 4.5.* and Python(3.5.* or 3.6.*).
Only supported with python runtime.
'''
# use an image available in public Container Registry without authentication
myenv.docker.base_image = "mcr.microsoft.com/azureml/o16n-sample-user-base/ubuntu-miniconda"

# or, use an image available in a private Container Registry
myenv.docker.base_image = "myregistry.azurecr.io/mycustomimage:1.0"
myenv.docker.base_image_registry.address = "myregistry.azurecr.io"
myenv.docker.base_image_registry.username = "username"
myenv.docker.base_image_registry.password = "password"

#######################################
#Step 6: Write the entry script
#######################################
#Write the script that will be used to predict on your model
'''
Create a “score.py” script:
Define the initialization method to load the registered model using JobLib.
Define the “run” method to:
Parse input data (JSON to NumPy array).
Predict using the model.
Return predictions in list form.
Handle exceptions effectively.
'''
%%writefile score.py
import os
import pickle
import json
import numpy
from sklearn.externals import joblib
from sklearn.linear_model import Ridge

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'sklearn_regression_model.pkl')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = numpy.array(data)
        result = model.predict(data)
        # you can return any data type as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error

#######################################
#Step 7: Create inference configuration
#######################################
'''
Use the “InferenceConfig” class to specify the entry script and environment.
Run the provided cells to finalize the inference configuration.
'''
from azureml.core.model import InferenceConfig
inf_config = InferenceConfig(entry_script='score.py', environment=myenv)

#Model Profiling
'''
Profile your model to understand how much CPU and memory the service, created as a result of its 
deployment, will need.
In order to profile your model you will need:

a registered model
an entry script
an inference configuration
a single column tabular dataset, where each row contains a string representing sample request 
data sent to the service.
'''
import json
from azureml.core import Datastore
from azureml.core.dataset import Dataset
from azureml.data import dataset_type_definitions

dataset_name='sample_request_data'

dataset_registered = False
try:
    sample_request_data = Dataset.get_by_name(workspace = ws, name = dataset_name)
    dataset_registered = True
except:
    print("The dataset {} is not registered in workspace yet.".format(dataset_name))

if not dataset_registered:
    input_json = {'data': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]}
    # create a string that can be put in the body of the request
    serialized_input_json = json.dumps(input_json)
    dataset_content = []
    for i in range(100):
        dataset_content.append(serialized_input_json)
    sample_request_data = '\n'.join(dataset_content)
    file_name = "{}.txt".format(dataset_name)
    f = open(file_name, 'w')
    f.write(sample_request_data)
    f.close()

    # upload the txt file created above to the Datastore and create a dataset from it
    data_store = Datastore.get_default(ws)
    data_store.upload_files(['./' + file_name], target_path='sample_request_data')
    datastore_path = [(data_store, 'sample_request_data' +'/' + file_name)]
    sample_request_data = Dataset.Tabular.from_delimited_files(
        datastore_path,
        separator='\n',
        infer_column_types=True,
        header=dataset_type_definitions.PromoteHeadersBehavior.NO_HEADERS)
    sample_request_data = sample_request_data.register(workspace=ws,
                                                    name=dataset_name,
                                                    create_new_version=True)

'''
Now that we have an input dataset we are ready to go ahead with profiling. In this case we are 
testing the previously introduced sklearn regression model on 1 CPU and 0.5 GB memory. 
The memory usage and recommendation presented in the result is measured in Gigabytes. 
The CPU usage and recommendation is measured in CPU cores.
'''
from datetime import datetime
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import Model, InferenceConfig


environment = Environment('my-sklearn-environment')
environment.python.conda_dependencies = CondaDependencies.create(conda_packages=[
    'pip==20.2.4'],
    pip_packages=[
    'azureml-defaults',
    'inference-schema[numpy-support]',
    'joblib',
    'numpy',
    'scikit-learn==0.22.1',
    'scipy'
])
inference_config = InferenceConfig(entry_script='score.py', environment=environment)
# if cpu and memory_in_gb parameters are not provided
# the model will be profiled on default configuration of
# 3.5CPU and 15GB memory
profile = Model.profile(ws,
            'sklearn-%s' % datetime.now().strftime('%m%d%Y-%H%M%S'),
            [model],
            inference_config,
            input_dataset=sample_request_data,
            cpu=1.0,
            memory_in_gb=0.5)

# profiling is a long running operation and may take up to 25 min
profile.wait_for_completion(True)
details = profile.get_details()

#############################################################
#Step 8: Provision the Azure Kubernetes Services cluster
#############################################################
'''
This is a one time setup. You can reuse this cluster for multiple deployments after it has been 
created. If you delete the cluster or the resource group that contains it, then you would have 
to recreate it.
'''
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your AKS cluster
aks_name = 'my-aks-9' 

# Verify that cluster does not exist already
try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # Use the default configuration (can also provide parameters to customize)
    prov_config = AksCompute.provisioning_configuration()

    # Create the cluster
    aks_target = ComputeTarget.create(workspace = ws, 
                                    name = aks_name, 
                                    provisioning_configuration = prov_config)

if aks_target.get_status() != "Succeeded":
    aks_target.wait_for_completion(show_output=True)

#######################################
#Step 9: Deploy the model
#######################################
#Deploy web service to AKS
# Set the web service configuration (using default here)
aks_config = AksWebservice.deploy_configuration()

# # Enable token auth and disable (key) auth on the webservice
# aks_config = AksWebservice.deploy_configuration(token_auth_enabled=True, auth_enabled=False)
%%time
aks_service_name ='aks-service-1'

aks_service = Model.deploy(workspace=ws,
                           name=aks_service_name,
                           models=[model],
                           inference_config=inf_config,
                           deployment_config=aks_config,
                           deployment_target=aks_target)

aks_service.wait_for_deployment(show_output = True)
print(aks_service.state)

#Test the web service using run method
%%time
import json

test_sample = json.dumps({'data': [
    [1,2,3,4,5,6,7,8,9,10], 
    [10,9,8,7,6,5,4,3,2,1]
]})
test_sample = bytes(test_sample,encoding = 'utf8')

prediction = aks_service.run(input_data = test_sample)
print(prediction)

#######################################
#Step 10: Clean up resources
#######################################
#Delete the service, image and model.
%%time
aks_service.delete()
model.delete()


















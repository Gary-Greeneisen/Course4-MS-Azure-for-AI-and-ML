###
#Filename -  Deploying Trained Models.py
###

'''
In this activity, we’ll guide you through transforming your trained model into a fully operational 
service on Azure Machine Learning, so it’s ready to provide tangible value to end-users and 
applications.

By the end of this activity, you will be able to:
Configure and navigate the Azure Machine Learning workspace to manage and deploy machine learning models.
Deploy a trained model as a scalable service by creating an inference configuration and selecting
appropriate Azure compute resources.
Test and validate the deployed model endpoint, ensuring reliable predictions for real-world 
applications.

Step-by-step guide to model deployment
This reading will guide you through the following steps:

Step 1: Access your workspace.
Step 2: Open the deployment example.
Step 3: Initialize your environment.
Step 4: Train and register your model.
Step 5: Set up a custom environment.
Step 6: Deploy the model.
Step 7: Clean up deployment.
'''

#Step 1: Access your workspace
#Navigate to https://ml.azure.com and sign in if prompted.

#Step 2: Open the deployment example
#Navigate to SDK v1 > how-to-use-azureml > deployment > deploy-to-cloud.
#Open the “model-register-and-deploy.ipynb” notebook.

#Step 3: Initialize your environment
#Ensure you are using the Python 3.8 Azure ML kernel.
import azureml.core

# Check core SDK version number.
print('SDK version:', azureml.core.VERSION)

#Initialize workspace
#Create a Workspace object from your persisted configuration.
from azureml.core import Workspace

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

#Step 4: Train and register your model
'''
Use Scikit-Learn to train a small model on the diabetes dataset:
Run the code to train the model.
'''
import dill

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

dataset_x, dataset_y = load_diabetes(return_X_y=True)
model = Ridge().fit(dataset_x, dataset_y)
dill.dump(model, open('sklearn_regression_model.pkl', 'wb'))

#Register input and output datasets
#Here, you will register the data used to create the model in your workspace.
import numpy as np

from azureml.core import Dataset


np.savetxt('features.csv', dataset_x, delimiter=',')
np.savetxt('labels.csv', dataset_y, delimiter=',')

datastore = ws.get_default_datastore()
datastore.upload_files(files=['./features.csv', './labels.csv'],
                       target_path='sklearn_regression/',
                       overwrite=True)

input_dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, 'sklearn_regression/features.csv')])
output_dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, 'sklearn_regression/labels.csv')])

#Register model
#Register a file or folder as a model by calling Model.register().
import sklearn

from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration


model = Model.register(workspace=ws,
                       model_name='my-sklearn-model',                # Name of the registered model in your workspace.
                       model_path='./sklearn_regression_model.pkl',  # Local file to upload and register as a model.
                       model_framework=Model.Framework.SCIKITLEARN,  # Framework used to create the model.
                       model_framework_version=sklearn.__version__,  # Version of scikit-learn used to create the model.
                       sample_input_dataset=input_dataset,
                       sample_output_dataset=output_dataset,
                       resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                       description='Ridge regression model to predict diabetes progression.',
                       tags={'area': 'diabetes', 'type': 'regression'})

print('Name:', model.name)
print('Version:', model.version)

#Step 5: Set up a custom environment
'''
Avoid using the default environment to ensure compatibility and security.
Create a custom environment with specific package versions, including:
“pip,” “Azure Machine Learning Defaults,” “Inferred Schema,” “Joblib,” and specific versions of “DIL,” “NumPy,” and “Scikit-Learn.”
Define the “score.py” script:
Include methods for initialization and running the model.
Example: “model.predict” is used to process input data and return predictions.
'''

#Step 6: Deploy the model
#Deploy the model as a web service on Azure Container Instances (ACI) or Azure Kubernetes Service (AKS).
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies


environment = Environment('my-sklearn-environment')
environment.python.conda_dependencies = CondaDependencies.create(conda_packages=[
    'pip==20.2.4'],
    pip_packages=[
    'azureml-defaults',
    'inference-schema[numpy-support]',
    'joblib',
    'dill==0.3.6',
    'numpy==1.23',
    'scikit-learn=={}'.format(sklearn.__version__)

#When using a custom environment, you must also provide Python code for initializing and running
# your model. An example script is included with this notebook.
with open('score.py') as f:
    print(f.read())

'''
Deploy your model in the custom environment by providing an InferenceConfig object to 
Model.deploy(). In this case we are also using the AciWebservice.deploy_configuration() method to
generate a custom deploy configuration.

Note: This step can take several minutes.
'''
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice


service_name = 'my-custom-env-service'

inference_config = InferenceConfig(entry_script='score.py', environment=environment)
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_config,
                       overwrite=True)
service.wait_for_deployment(show_output=True)

#After your model is deployed, make a call to the web service using service.run().
input_payload = json.dumps({
    'data': dataset_x[0:2].tolist()
})

output = service.run(input_payload)

print(output)

#Step 7: Clean up deployment
'''
Use “service.delete” to remove the deployed service.
Cleaning up ensures that unnecessary charges are avoided and resources are freed.
'''
service.delete()






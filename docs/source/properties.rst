Properties Description
======================

This documentation describes each property in the example_properties.yaml file (in app/resources folder).

Config section
--------------

The config parameter contains configuration for notification emails as well as connection details for the 2 ElasticSearch
instances used in the project. For notification emails, a sender, a receiver and a password should be provided. The configuration
used in the project is for Gmail server. Regarding the ElasticSearch instances, elastic_retrieve configuration is related
to the ElasticSearch instance in SWO, from which articles are retrieved in order to execute the DebateLab pipeline. Finally,
the elastic_save is the instance where the processed documents are saved, containing ADU and relation information.

Tasks section
-------------

In the tasks field, a list with the tasks should be provided. The available options are:

* prep, for data preprocessing
* train, to train the Argument Mining models
* eval, to run the DebateLab pipeline
* error, to run error analysis

Preprocessing step
------------------

For each one of the aforementioned task options there are more specific configurations to be provided. Regarding the
preprocessing step, you can select to preprocess any of the available datasets, i.e. kasteli or essays; one of those
values should be given in the dataset field. Also, the project supports oversampling on the selected dataset and
you can select the number of instances for each class. Take a look at the oversampling section in the example_properties.yaml

Training step
-------------

Next, in the train section you can provide a list with the models you want to train. The available options are:

* adu, to train a model to predict ADU segments
* rel, to train a model to predict relations between ADUs
* stance, to train a model to predict the stance of a claim towards a major claim
* sim, to train a model to predict similarity between arguments
* alignment, to train a student model and align its embeddings for english/greek

For each one of these options, you should provide the respective section with the model parameters. For the adu model,
check the seq_model section, for the rel, stance and sim models check the class_model section and finally for the
alignment model check the alignment section in the example_properties.yaml to find out all the available configuration
options.

Pipeline step
-------------

The last configuration needed is regarding the eval step (DebateLab pipeline). The model field refers to the .pt files
that flair library creates; you can use either the best.pt or the final.pt models. The retrieve field defines the way
that articles should be retrieved from the ElasticSearch in SWO. For testing / dev purposes, the file option is used; where
links are read from a file and then retrieved from the ElasticSearch. In the final version of the project, the documents
will be retrieved through a query based on the date. Therefore, the last_date field will be automatically updated after each run.
The ner_endpoint field points to the API which extract Named Entities from a document. The max_correction_tries is used
for the number of validation runs that will be performed for each document. Finally, the notify section refers to the details
to notify ICS; the url and the credentials should be provied for the API call.
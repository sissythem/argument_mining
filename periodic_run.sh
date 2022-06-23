#!/bin/env bash
srcdir="/code"
python_exec="/usr/local/bin/python"

echo "Running periodic script at $(date)"


cd "$srcdir"
if [ ! -f "pipeline.py" ]; then
  echo "Can't find pipeline.py!" ; exit 1;
fi

adu_model="/models/adu"
rel_model="/models/rel"
stance_model="/models/stance"
echo "Runing pipeline"
$python_exec pipeline.py  \
        -adu_model "${adu_model}" \
        -rel_model "${rel_model}" \
        -stance_model "${stance_model}" \
        -elastic_config elastic_config_periodic.json \
        -input_type elastic \
        --newline_norm  \
        -topics_config "{\"model\": \"nlpaueb/bert-base-greek-uncased-v1\", \"n_clusters\": 32, \"device\": \"cpu\"}" \
        -ner_config "{\"endpoint\": \"http://petasis.dyndns.org:7100/predict-ner\"}" \
        -cross_doc_clustering_config "{\"model_name\": \"nlpaueb/bert-base-greek-uncased-v1\",\"clustering\": \"similarity\"}" \
        --run_cross_doc_clustering \
        --use_additional_cross_doc_inputs \
        --run_topics \
        --save_to_elastic \
        -notify_ics_credentials "$srcdir/notify_credentials.json"
echo "Pipeline run complete."

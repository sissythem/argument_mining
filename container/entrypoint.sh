project_root="/code"
git checkout HEAD^1 && git pull origin develop
mount_dir="/models"

[[ -z $PYTHON_EXEC ]] && PYTHON_EXEC="/usr/local/bin/python"

# test ellogon
# sources:
cd /opt/Ellogon
[[ -z $ELLOGON_VERSION ]] && ELLOGON_VERSION="152"
current_version="$(svn info --show-item revision)" 
if [[ "${current_version}" == "${ELLOGON_VERSION}" ]]; then
  echo "Validated ellogon version ${current_version}"
else
  echo "Expected ellogon version [${ELLOGON_VERSION}] but found version [${current_version}]!"
  exit 1
fi

# package:
cd $PYTHON_ELLOGON_PATH/ellogon 
echo "Checking ellogon python sources at [$(pwd)]"
# we can't check version -- make a naive source check
if [ $(grep 'ExtendedUnicode' tokeniser.py | wc -l ) == 3 ]; then
  echo "Validated ellogon python package"
else
  echo "Unexpected ellogon python source contents -- e.g. tokeniser.py : $(ls; head tokeniser.py)"
  exit 1
fi

cd "${project_root}"

[[ -z $PERIODIC_CONTAINER ]] && export PERIODIC_CONTAINER="debatelab-periodic"
[[ -z $PERIODIC_PORT ]] && export PERIODIC_PORT=8001

echo "Running container deployment in [${RUN_MODE}] mode."
echo "Running from $(pwd) with python exec: [$PYTHON_EXEC]"



# move keys / crawler creds to source root
sensitive_files=("authorized_keys" "crawler_credentials.txt" "notify_credentials.json")
for sfile in ${sensitive_files[@]} ; do 
  sfilepath="${mount_dir}/${sfile}"
  if [ -f "${sfilepath}" ]; then 
    echo "Copying ${sfilepath} to project root: ${project_root}."
    cp "${sfilepath}" "${project_root}"
  else
    echo "Missing $sfile at mounted volume at: $mount_dir !"
  fi
done

if [ "$RUN_MODE" == "rest" ]; then

  
  $PYTHON_EXEC deploy.py -pipeline_configuration container/configuration_rest.json \
    -crawler_credentials_path "$project_root/crawler_credentials.txt" \
    -write_crossdocs_to "url" \
    -models_base_path "/models" \
    -crossdoc_output_path "http://${PERIODIC_CONTAINER}:${PERIODIC_PORT}/crossdocs"

elif [  "$RUN_MODE" == "annotation"  ]; then

  $PYTHON_EXEC deploy.py -pipeline_configuration container/configuration_annotation.json \
    -crawler_credentials_path "$project_root/crawler_credentials.txt"  \
    -models_base_path "/models" \
    --allow_multiple_models \
    -disable_authentication
elif [  "$RUN_MODE" == "periodic"  ]; then
  # start cron and register crontab
  service cron start
  crontab /code/periodic.cron
  # deploy API just for ingesting additional docs for cross-document clustering
  $PYTHON_EXEC deploy.py -pipeline_configuration "" \
    -models_base_path "/models" \
    -crawler_credentials_path "" \
    -write_crossdocs_to "disk" \
    -port "${PERIODIC_PORT}"

else
  echo "Undefined run mode: [${RUN_MODE}]"
fi

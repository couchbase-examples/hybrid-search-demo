## Hybrid Movie Search using Couchbase

This is a demo app built to perform hybrid search using the Vector Search capabilities of Couchbase. The demo allows users to search for movies based on the synopsis or overview of the movie using both the native [Couchbase Python SDK](https://docs.couchbase.com/python-sdk/current/howtos/full-text-searching-with-sdk.html) and using the [LangChain Vector Store integration](https://python.langchain.com/docs/integrations/vectorstores/couchbase/).

> Note that you need Couchbase Server 7.6 or higher for Vector Search and 7.6.4 or higher for pre-filter support.

### How does it work?

You can perform semantic searches for movies based on the plot synopsis. Additionally, you can filter the results based on the year of release and the IMDB rating for the movie. Optionally, you can also search for the keyword in the movie title.

![hybrid search demo](hybrid_search_demo.png)

The hybrid search can be performed using both the Couchbase Python SDK & the LangChain Vector Store integration for Couchbase. We use OpenAI for generating the embeddings.

### How to Run

- #### Install dependencies

  `pip install -r requirements.txt`

- #### Set the environment secrets

  Copy the `secrets.example.toml` file and rename it to `secrets.toml` and replace the placeholders with the actual values for your environment.

  > For the ingestion script, the same environment variables need to be set in the environment (using `.env` file from `.env.example`) as it runs outside the Streamlit environment.

  ```
  OPENAI_API_KEY = "<open_ai_api_key>"
  DB_CONN_STR = "<connection_string_for_couchbase_cluster>"
  DB_USERNAME = "<username_for_couchbase_cluster>"
  DB_PASSWORD = "<password_for_couchbase_cluster>"
  DB_BUCKET = "<name_of_bucket_to_store_documents>"
  DB_SCOPE = "<name_of_scope_to_store_documents>"
  DB_COLLECTION = "<name_of_collection_to_store_documents>"
  INDEX_NAME = "<name_of_search_index_with_vector_support>"
  EMBEDDING_MODEL = "text-embedding-3-small" # OpenAI embedding model to use to encode the documents
  ```

- #### Create the Search Index on Full Text Service

  We need to create the Search Index on the Full Text Service in Couchbase. For this demo, you can import the following index using the instructions.

  - [Couchbase Capella](https://docs.couchbase.com/cloud/search/import-search-index.html)

    - Copy the index definition to a new file index.json
    - Import the file in Capella using the instructions in the documentation.
    - Click on Create Index to create the index.

  - [Couchbase Server](https://docs.couchbase.com/server/current/search/import-search-index.html)

    - Click on Search -> Add Index -> Import
    - Copy the following Index definition in the Import screen
    - Click on Create Index to create the index.

  #### Index Definition

  Here, we are creating the index `movies-search-demo` on the documents in the `_default` collection within the `_default` scope in the bucket `movies`. The Vector field is set to `Overview_embedding` with 1536 dimensions and the text field set to `Overview`. We are also indexing and storing some of the other fields in the document for the hybrid search. The similarity metric is set to `dot_product`. If there is a change in these parameters, please adapt the index accordingly.

  ```json
  {
    "type": "fulltext-index",
    "name": "movies._default.movies-search-demo",
    "uuid": "7103dcd1a3781f50",
    "sourceType": "gocbcore",
    "sourceName": "movies",
    "planParams": {
      "maxPartitionsPerPIndex": 64,
      "indexPartitions": 16
    },
    "params": {
      "doc_config": {
        "docid_prefix_delim": "",
        "docid_regexp": "",
        "mode": "scope.collection.type_field",
        "type_field": "type"
      },
      "mapping": {
        "analysis": {},
        "default_analyzer": "standard",
        "default_datetime_parser": "dateTimeOptional",
        "default_field": "_all",
        "default_mapping": {
          "dynamic": false,
          "enabled": false
        },
        "default_type": "_default",
        "docvalues_dynamic": false,
        "index_dynamic": false,
        "store_dynamic": false,
        "type_field": "_type",
        "types": {
          "_default._default": {
            "dynamic": false,
            "enabled": true,
            "properties": {
              "IMDB_Rating": {
                "dynamic": false,
                "enabled": true,
                "fields": [
                  {
                    "index": true,
                    "name": "IMDB_Rating",
                    "store": true,
                    "type": "number"
                  }
                ]
              },
              "Overview": {
                "dynamic": false,
                "enabled": true,
                "fields": [
                  {
                    "analyzer": "en",
                    "index": true,
                    "name": "Overview",
                    "store": true,
                    "type": "text"
                  }
                ]
              },
              "Overview_embedding": {
                "dynamic": false,
                "enabled": true,
                "fields": [
                  {
                    "dims": 1536,
                    "index": true,
                    "name": "Overview_embedding",
                    "similarity": "dot_product",
                    "type": "vector",
                    "vector_index_optimized_for": "recall"
                  }
                ]
              },
              "Poster_Link": {
                "dynamic": false,
                "enabled": true,
                "fields": [
                  {
                    "analyzer": "en",
                    "index": true,
                    "name": "Poster_Link",
                    "store": true,
                    "type": "text"
                  }
                ]
              },
              "Released_Year": {
                "dynamic": false,
                "enabled": true,
                "fields": [
                  {
                    "index": true,
                    "name": "Released_Year",
                    "store": true,
                    "type": "number"
                  }
                ]
              },
              "Runtime": {
                "dynamic": false,
                "enabled": true,
                "fields": [
                  {
                    "analyzer": "en",
                    "index": true,
                    "name": "Runtime",
                    "store": true,
                    "type": "text"
                  }
                ]
              },
              "Series_Title": {
                "dynamic": false,
                "enabled": true,
                "fields": [
                  {
                    "analyzer": "keyword",
                    "index": true,
                    "name": "Series_Title",
                    "store": true,
                    "type": "text"
                  }
                ]
              }
            }
          }
        }
      },
      "store": {
        "indexType": "scorch",
        "segmentVersion": 16
      }
    },
    "sourceParams": {}
  }
  ```

- #### Ingest the Documents

  For this demo, we are using the [IMDB dataset from Kaggle](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows). You can download the CSV file, `imdb_top_1000.csv` to the source folder or use the one provided in the repo.

  To ingest the documents including generating the embeddings for the Overview field, you can run the script, `ingest.py`

  `python ingest.py`

- #### Run the application

  `streamlit run hybrid_search.py`

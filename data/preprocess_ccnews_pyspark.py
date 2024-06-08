"""
EMR Step function configuration:
Step type: custom JAR
JAR location: command-runner.jar (this is a default JAR, no need to upload anything)
Arguments: (do not add backslashes `\` into the config)
spark-submit --deploy-mode cluster --conf spark.driver.memory=96G --conf spark.executor.memory=70G
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer --conf spark.kryoserializer.buffer.max=2000M
--conf spark.driver.maxResultSize=0 --conf spark.jars.packages=com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.8
s3://.../preprocess_pyspark.py
--input_path s3://.../cc_news_202101/*.jsonl
--output_path s3://.../spark-outputs
--date_from "2021-02-01 0:0:0" --date_to "2021-02-01 1:0:0"
"""
import argparse
import re
import spacy
import pyspark.sql.functions as f
from datetime import datetime
from pyspark.sql.functions import col
from pyspark.sql.functions import sha2, concat_ws
from pyspark.sql import SparkSession
import sparknlp
from pyspark.sql.functions import when
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

from pyspark.ml import PipelineModel
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.types import *
import pandas as pd

RE_START_DASH = [r'()– ']

RE_NEWSPAPER = r'(?:[Tt]he )?(?:St\. )?(?:[A-Z][A-z]+[ -.]){0,3}(?:of )?(?:[A-Z][A-z]+)(?: \d{1,2})?(?:\'s)?(?:\.com|\.edu)?'

RE_NAME = r'[A-Z][a-z]+ [A-Z][a-z]+(?:-[A-Z][a-z]+)?'
RE_NAMES = r'{}(?: and {})?'.format(RE_NAME, RE_NAME)

RE_REPORT = [r',("?) {} (?:reports|notes|finds|quips|writes|explains|said)(?=[\.,])'.format(RE_NEWSPAPER),
             r',("?) (?:reports|notes|finds|quips|writes|explains) {}(?=[\.,])'.format(RE_NEWSPAPER),
             r',("?) according to a (?:new )? report by {}(?=[\.,])'.format(RE_NEWSPAPER),
             r',("?) according to a (?:new )? (?:report|press release) by {}(?=[\.,])'.format(RE_NEWSPAPER),
             r',("?) according to {}(?=[\.,])'.format(RE_NEWSPAPER),
             r',("?) per {}(?=[\.,])'.format(RE_NEWSPAPER),
             r',”( )(?:s)?he writes|explains in the {}, “'.format(RE_NEWSPAPER),
             ]
RE_REMOVE_ALL = [re.compile(r) for r in RE_START_DASH + RE_REPORT]

BAD_LAST_SENT_START = ['Click here ', 'Click for ', 'Click through ', 'Click to read ', 'Read more ', 'More details ',
                       'Read the full ', 'For more ', '(For more ', 'Head to the ', 'Read ']
BAD_LAST_SENT_END = [' here.', 'here.)']


def load_spark():
    spark = SparkSession.builder \
        .appName("Spark NLP") \
        .config("spark.driver.memory", "128") \
        .config("spark.executor.memory", "96G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12") \
        .config("spark.maximizeResourceAllocation", "true") \
        .config("spark.executor.cores", "32") \
        .config("spark.default.parallelism", "2000") \
        .config("spark.executor.instances", "8") \
        .getOrCreate()
    return spark


def load_data(spark, input_path):
    df = spark.read.option("multiline", "true").json(input_path)
    return df


def filter_time_range(df, date_from, date_to):
    date_from_object = datetime.strptime(date_from, '%Y-%m-%d %H:%M:%S')
    date_to_object = datetime.strptime(date_to, '%Y-%m-%d %H:%M:%S')
    df = df.where(col('date_publish').between(date_from_object, date_to_object))
    return df


def filter_language_and_domain(df):
    exclude_domains = [
        # not English
        "hindi.news18.com",
        "ictmarketexperts.com",
        # financial news
        "www.americanbankingnews.com",
        "www.tickerreport.com",
        "www.thestreet.com",
        "www.themarketsdaily.com"
    ]
    df = df.filter((df["language"] == "en"))
    df = df.filter(~df.source_domain.isin(exclude_domains))
    return df


def create_sha2_hash(df):
    df = df.withColumn("id", sha2(df.maintext, 256))
    return df


def dedup(df):
    df = df.dropDuplicates(["maintext"])
    df = df.withColumn('maintext_sub200char', df.maintext.substr(1, 200))
    df = df.dropDuplicates(["title", "maintext_sub200char"])
    return df


def filter_by_word_count(df, maintext_word_limit=50, title_word_limit=(5, 25)):
    df = df.withColumn('wordCountTitle', f.size(f.split(f.col('title'), ' ')))
    df = df.withColumn('wordCountMaintext', f.size(f.split(f.col('maintext'), ' ')))
    df = df.filter((df["wordCountMaintext"] >= maintext_word_limit))
    df = df.filter((df["wordCountTitle"] >= title_word_limit[0]) & (df["wordCountTitle"] < title_word_limit[1]))
    return df


def extract_lead_sentence(df):
    documenter = DocumentAssembler() \
        .setInputCol("maintext") \
        .setOutputCol("document")

    sentencerDL = SentenceDetectorDLModel \
        .pretrained("sentence_detector_dl", "en") \
        .setInputCols(["document"]) \
        .setOutputCol("sentences")

    sentence_detector_pipeline = PipelineModel(stages=[documenter, sentencerDL])
    df = sentence_detector_pipeline.transform(df)
    df = df.withColumn("lead_sentence", df.sentences[0])
    df = df.withColumn('lead_sentence_trailing', f.split(f.col('lead_sentence').result, '\n')) \
        .withColumn("lead_sentence_trailing", f.col("lead_sentence_trailing")[f.size("lead_sentence_trailing") - 1])
    return df


def remove_brackets(df):
    df = df.withColumn(
        'maintext',
        f.regexp_replace(f.col('maintext'), "[\s]\((.+?)\)", ''))
    return df


def clean_bylines(df):
    regexes = [
        u".*[a-zA-Z\s|\,|\.|\(|\)\[\]]+?[\s]?[\u2013|\u2014|\-|\:]+[\s]+(?![a-z|\s]+)",
        # dashes: e.g., 'ATLANTIC CITY, N.J. (AP) - ', must be followed by capital letters
        u"^[A-Z|\,|\.|\(|\)\s]*[\u2013|\u2014|\-|\u003A|>\u2022]+[\s]+",
        # ALL CAPITAL followed by dash or colon or •: e.g., 'NEW DELHI-'
        "^.*[\(\[][A-Z]*[\)\]]\s(?![a-z|\s]+)",  # e.g., "Dehradun/New Delhi, Feb 7 (PTI) [followed by capital letters]"
        u"^[a-zA-Z\s\,]*\u003A\s(?![a-z|\s]+)",  # e.g., "Washington: Suspected ", "[NFA] Captain Tom Moore"
    ]
    df = df.withColumn('lead_sentence_cleaned', f.col('lead_sentence_trailing'))
    for reg in regexes:
        df = df.withColumn(
            'lead_sentence_cleaned_new',
            f.regexp_replace(f.col('lead_sentence_cleaned'), reg, ''))
        df = df.withColumn(
            'regex_diff',
            f.length('lead_sentence_cleaned') -
            f.length('lead_sentence_cleaned_new'))
        df = df.withColumn("lead_sentence_cleaned",
                           when(df["regex_diff"] < 50, df["lead_sentence_cleaned_new"]).otherwise(
                               df["lead_sentence_cleaned"]))
    return df


@udf(StringType())
def clean_sentence(sent):
    sent = re.sub(
        r',(["”]?) (?:{}) (?:reports|notes|finds|quips|writes|explains)( in the {})?\.$'.format(RE_NAMES, RE_NEWSPAPER),
        r'.\1', sent)
    sent = re.sub(r',(["”]?) (?:{}) (?:reports|notes|finds|quips|writes|explains)\.$'.format(RE_NEWSPAPER), r'.\1',
                  sent)
    sent = re.sub(r'^{} reports on '.format(RE_NEWSPAPER), 'Articles report on ', sent)
    sent = re.sub(r'^{} (?:reports|notes|finds|writes|explains) that ([A-z])'.format(RE_NEWSPAPER),
                  lambda x: x.group(1).capitalize(), sent)
    sent = re.sub(r'^{} (?:reports|notes|finds|writes|explains) ([A-z])'.format(RE_NEWSPAPER),
                  lambda x: x.group(1).capitalize(), sent)
    sent = re.sub(r'^As {} (?:reports|notes|finds|writes|explains) ([a-z])'.format(RE_NEWSPAPER),
                  lambda x: x.group(1).capitalize(), sent)
    sent = re.sub(r'^According to {}, ([a-z])'.format(RE_NEWSPAPER), lambda x: x.group(1).capitalize(), sent)
    sent = re.sub(r',("?) according to .*?(?=[\.,])', lambda x: x.group(1).capitalize(), sent)
    sent = re.sub(r',("?)[^,]*said(?=[\.,])', lambda x: x.group(1).capitalize(), sent)
    return sent


def clean_sources(df):
    df = df.withColumn('lead_sentence_cleaned', clean_sentence(f.col('lead_sentence_cleaned')))
    return df


def load_spacy_model():
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
    nlp.add_pipe('sentencizer')
    return nlp

def entities_from_nlp(column, broadcasted_nlp):
    @pandas_udf(ArrayType(ArrayType(StringType())))
    def entities(list_of_text: pd.Series) -> pd.Series:
        # retrieving the shared nlp object
        nlp = broadcasted_nlp.value
        # batch processing our list of text
        docs = nlp.pipe(list_of_text)

        ents = [[[ent.text, ent.label_] for ent in doc.ents] for doc in docs]

        return pd.Series(ents)
    return entities(column)


@pandas_udf(ArrayType(StringType()))
def entity_text(nested_list: pd.Series) -> pd.Series:
    entity_types = [[i[0] for i in row] for row in nested_list]

    return pd.Series(entity_types)


@pandas_udf(ArrayType(StringType()))
def entity_type(nested_list: pd.Series) -> pd.Series:
    entity_types = [[i[1] for i in row] for row in nested_list]

    return pd.Series(entity_types)


def spacy_ner(df, broadcasted_nlp):
    df = df.withColumn('spacy_lead', entities_from_nlp('lead_sentence_cleaned', broadcasted_nlp))
    df = df.withColumn('spacy_maintext', entities_from_nlp('maintext', broadcasted_nlp))

    df = df.withColumn('lead_entity_type', entity_type('spacy_lead'))
    df = df.withColumn('lead_entity_text', entity_text('spacy_lead'))
    df = df.withColumn('maintext_entity_type', entity_type('spacy_maintext'))
    df = df.withColumn('maintext_entity_text', entity_text('spacy_maintext'))

    df = df.withColumn('entity_count_lead', f.size(col('lead_entity_text')))
    df = df.withColumn('entity_count_maintext', f.size(col('maintext_entity_type')))

    return df


def save(df, output_path, date_from, date_to):
    df_save = df.select('id', 'authors', 'date_publish', 'title', 'maintext',
                        'description', 'source_domain', 'url', 'wordCountTitle',
                        'wordCountMaintext', 'lead_sentence_cleaned',
                        'lead_entity_type', 'lead_entity_text', 'maintext_entity_type', 'maintext_entity_text',
                        'entity_count_lead', 'entity_count_maintext'
                        #                         'lead_entity_text'
                        #                         'spacy_lead', 'spacy_maintext'
                        )
    #     df_save = df_save.coalesce(50)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"{output_path}/{date_from}-{date_to}/{current_time}.parquet"
    df_save.write.parquet(save_path)
    return df_save


def main(spark, broadcasted_nlp, input_path, output_path, date_from, date_to):
    df = load_data(spark, input_path)
    df = filter_time_range(df, date_from, date_to)
    df = filter_language_and_domain(df)
    # #     df = df.repartition(1000)
    df = filter_by_word_count(df)
    df = dedup(df)
    df = create_sha2_hash(df)
    df = remove_brackets(df)
    df = extract_lead_sentence(df)
    df = clean_bylines(df)
    df = clean_sources(df)
    df = spacy_ner(df, broadcasted_nlp)
    save(df, output_path, date_from, date_to)


if __name__ == '__main__':
    spark = load_spark()
    broadcasted_nlp = spark.sparkContext.broadcast(load_spacy_model())
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help='input data in s3')
    parser.add_argument('--output_path', help='output data in s3')
    parser.add_argument('--date_from', default='2021-02-01 0:0:0', help='data to filter date from, format in %Y-%m-%d %H:%M:%S')
    parser.add_argument('--date_to', default='2021-02-01 1:0:0', help='data to filter date to, format in %Y-%m-%d %H:%M:%S')
    args = parser.parse_args()

    main(spark, broadcasted_nlp, args.input_path, args.output_path, args.date_from, args.date_to)

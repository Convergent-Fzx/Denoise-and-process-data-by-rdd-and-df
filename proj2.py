from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import re
import itertools


class Project2:
    
    def is_valid_word(self, word):
        return len(word) >= 1 and 'a' <= word[0] <= 'z' and word not in self.stopwords_broadcast.value

    def run(self, inputPath, outputPath, stopwordsPath, k):
        spark = SparkSession.builder.master("local").appName("project2_df").getOrCreate()
        
        stopwords_list = spark.read.text(stopwordsPath).rdd.map(lambda row: row[0].lower()).collect()
        self.stopwords_broadcast = spark.sparkContext.broadcast(set(stopwords_list))

        lines = spark.sparkContext.textFile(inputPath)
        lines = lines.map(lambda line: (line.split(',')[0].lower(), [word.lower() for word in line.split(',')[1:]]))

        validLines = lines.map(
            lambda line: (
                line[0], 
                [word for word in re.split("[\\s*$&#/\"'\\,.:;?!\\[\\](){}<>~\\-_]+", ' '.join(line[1])) if self.is_valid_word(word)]
            )
        )
        
        '''非法行计算'''
        invalid_lines = validLines.map(lambda x: (x[0], (1, len(x[1])))).filter(lambda x: x[1][1] < 3)
        valid_lines = validLines.filter(lambda x: len(x[1]) >= 3)
        valid_lines_DF = valid_lines.map(lambda x: Row(first_word=x[0], word=sorted(x[1]))).toDF()

        res_inv = invalid_lines.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        res_inv = res_inv.map(lambda x:(x[0],x[1][0]))
        res_inv = res_inv.sortByKey()
        res_inv_DF = res_inv.map(lambda x: Row(first_word=x[0], count=x[1])).toDF()

        def is_valid_word(word):
            return len(word) >= 1 and 'a' <= word[0] <= 'z' and word not in self.stopwords_broadcast.value
        is_valid_word_udf = udf(is_valid_word, BooleanType())

        '''zongDF'''
        linesDF = spark.read.text(inputPath).withColumnRenamed("value", "line")
        linesDF = linesDF.withColumn("first_word", lower(split(col("line"), ",")[0]))
        linesDF = linesDF.withColumn("remaining_words", expr("slice(split(line, ' '), 2, size(split(line, ' ')) - 1)"))  

        linesDF = linesDF.withColumn(
            "lower_remaining_words",
            expr("transform(remaining_words, x -> lower(x))")
        )

        zongDF = linesDF.groupBy("first_word").count()
        hegeDF = zongDF.join(res_inv_DF, zongDF.first_word == res_inv_DF.first_word, "outer") \
                       .select(zongDF.first_word, 
                               zongDF["count"].alias("total_count"), 
                               res_inv_DF["count"].alias("invalid_count")) \
                       .withColumn("vc", col("total_count") - coalesce(col("invalid_count"), lit(0)))
        '''cata,total,invalid,valid'''
        
        '''非法行输出'''
        res_inv_formatted = res_inv_DF.select(
            concat(
                col("first_word"),
                lit(" invalid line: "),
                col("count")
            ).alias("formatted_output")
        )

        '''copy'''
        hegecDF = hegeDF
        
        def generate_triplets(words):
            triplets = list(itertools.combinations(words, 3))
            return [tuple(sorted(triplet)) for triplet in triplets]  

        generate_triplets_udf = udf(generate_triplets, ArrayType(ArrayType(StringType())))

        # 生成三元组并展开
        tripletsDF = valid_lines_DF.withColumn("triplets", generate_triplets_udf(col("word")))

        explodedDF = tripletsDF.select(
            col("first_word").alias("fw"),
            explode(col("triplets")).alias("triplet")
        )

        # 过滤有效三元组
        validTripletsDF = explodedDF.filter(
            is_valid_word_udf(col("triplet")[0]) &
            is_valid_word_udf(col("triplet")[1]) &
            is_valid_word_udf(col("triplet")[2])
        )

        windowSpec = Window.partitionBy("fw").orderBy(col("count").desc())
        aggregatedDF = validTripletsDF.groupBy("fw", "triplet").count()
        aggregatedDF = aggregatedDF.withColumn("rank",row_number().over(Window.partitionBy("fw").orderBy(col("count").desc())))\
                                   .filter(col("rank")<=k).drop("rank")
        '''fw,sanyuanzu,bili'''
        
        k = int(k)  
        
        join_res = aggregatedDF.join(hegecDF, hegecDF.first_word == aggregatedDF.fw, "outer")
        join_res = join_res.withColumn("bili", col("count") / col("vc"))
        
        resultDF = join_res.select("fw", "triplet", "bili")
        res_inv_formatted = res_inv_DF.rdd.map(lambda x: "{}\tinvalid line:{}".format(x[0], x[1]))
        top_k_per_fw = resultDF.withColumn("rank", row_number().over(Window.partitionBy("fw").orderBy(col("bili").desc()))) \
                                .filter(col("rank") <= k) \
                                .drop("rank")

        formatted_res = top_k_per_fw.rdd.map(lambda x: "{}\t{}:{}".format(x[0], ','.join(x[1]), x[2]))
        
        combined_results = res_inv_formatted.union(formatted_res)
        combined_results = combined_results.sortBy(lambda x: x[0])

        top_k_per_fw.show(truncate=False)
        combined_results.coalesce(1).saveAsTextFile(outputPath)
        
        spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Wrong arguments")
        sys.exit(-1)
    Project2().run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
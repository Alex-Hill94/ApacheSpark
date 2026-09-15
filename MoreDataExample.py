from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("MoreData") \
    .master("local[*]") \
    .getOrCreate()

# Create a large DataFrame
df = spark.range(0, 1000000)

# Trigger a computation
values = df.filter(df.id % 3 == 0)
values = values.repartition(16)
# Explains logic of new DataFrame
values.explain()

# Prints DataFrame
values.show()

# Performs job
counts = values.count()

spark.stop

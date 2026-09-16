from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("SimpleExample") \
    .master("local[*]") \
#    .config("spark.driver.bindAddress", "127.0.0.1") \
#    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

# Create some data
data = [
    ("Alice", 25),
    ("Bob", 17),
    ("Charlie", 30),
    ("Diana", 15)
]

# Create a DataFrame
df = spark.createDataFrame(data, ["name", "age"])

# Filter adults
adults = df.filter(df.age >= 18)

# Display results
adults.show()

# Stop Spark
spark.stop()
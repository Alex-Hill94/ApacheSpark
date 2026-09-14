"""
PySpark "Hello, World!" example code.

Creates a local Spark session, builds a small DataFrame, and runs
a couple of basic transformations/actions to show PySpark working.
"""

from pyspark.sql import SparkSession


def main():
    # Create a local Spark session
    spark = SparkSession.builder \
        .appName("HelloWorld") \
        .master("local[*]") \
        .getOrCreate()

    # Create a simple DataFrame
    data = [("Hello",), ("World",), ("from",), ("PySpark",)]
    df = spark.createDataFrame(data, ["word"])

    # Simple operation #1: show the data
    print("Original data:")
    df.show()

    # Simple operation #2: count rows
    print(f"Number of rows: {df.count()}")

    # Simple operation #3: collect and join into a sentence
    words = [row["word"] for row in df.collect()]
    print(" ".join(words))

    spark.stop()


if __name__ == "__main__":
    main()

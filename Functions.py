from pyspark.sql import functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Functions") \
    .master("local[*]") \
    .getOrCreate()


# Creates the First DataFrame 
employees = spark.createDataFrame(
    [
        (1, "Alice", 10, 60000),
        (2, "Bob",   10, 50000),
        (3, "Cara",  20, 70000),
        (4, "Dan",   30, 45000),
    ],
    ["employee_id", "name", "department_id", "salary"]
)

# Creates a Second DataFrame with relevant data to the first
departments = spark.createDataFrame(
    [
        (10, "Engineering"),
        (20, "Sales"),
        (30, "HR"),
    ],
    ["department_id", "department_name"]
)


## 1. SELECTING AND TRANSFORMING COLUMNS ##
### 1.1 df.select ###

def ex1_1a():
    # (1.1a) Selects two columns based on header and prints the result
    print('1.1a')
    employees.select("name", "salary"
    ).show()

def ex1_1b():
    # (1.1b) Creates a new calculated column, note this does not change the DataFrame
    print('1.1b')
    employees.select(
        "name",
        "salary",
        (F.col("salary") * 1.1).alias("salary_with_raise")
    ).show()

def ex1_1c():
    ## Other functions within 'select'

    # F.col("salary") ----> reference a column
    # F.lit(10)        ----> literal value
    # F.alias(...)    ----> rename an expression
    # F.when(...).otherwise(...) ----> conditional expressing

    # (1.1c) Selects values from column 'name' and creates a new
    # column, salary_band, with conditional values
    print('1.1c')
    employees.select(
        "name",
        F.when(F.col("salary") >= 60000, "High")
        .otherwise("Standard")
        .alias("salary_band")
    ).show()


### 1.2 df.withColumn ###

def ex1_2a():
    # (1.2a) Creates new Dataframe based on employees DataFrame, but with a 
    # new column, bounus, added
    print('1.2a')
    employees_with_bonus = employees.withColumn(
        "bonus",
        F.col("salary") * 0.10
    )
    employees_with_bonus.show()

def ex1_2b():
    # (1.2b) Adds another column to the existing employees_with_bonus DataFrame
    print('1.2b')
    employees_with_bonus = employees.withColumn(
    "bonus",
    F.col("salary") * 0.10
    )

    employees_with_bonus = employees_with_bonus.withColumn(
        "total_compensation",
        F.col("salary") + F.col("bonus")
    )
    employees_with_bonus.show()

def ex1_2c():
    # (1.2c) Creates a new DataFrame like employees, but with a column renamed
    print('1.2c')
    employees_with_bonus = employees.withColumn(
        "bonus",
        F.col("salary") * 0.10
    )
    renamed = employees.withColumnRenamed(
        "salary",
        "annual_salary"
    )
    renamed.show()

    # Removes column
    renamed.drop("employee_id").show()


#### 2. FILTERING ROWS ####
def ex2a():
    # (2a) Filters DataFrame only based on one condition
    print('2a')
    employees.filter(F.col("salary") >= 60000
    ).show()

def ex2b():
    # (2a) Filters DataFrame only based on multiple conditions
    print('2b')
    employees.filter((F.col("salary") >= 60000) 
                    & (F.col("department_id") == 10)
    ).show()


def ex3():
    # (3) Groups data in terms of unique department_ids, and computes
    # aggregate statistics for these based on the 'salary' columm
    print('3')
    summary = employees.groupBy("department_id").agg(
        F.count("*").alias("employee_count"),
        F.avg("salary").alias("average_salary"),
        F.min("salary").alias("minimum_salary"),
        F.max("salary").alias("maximum_salary"),
        F.sum("salary").alias("total_salary")
    ).show()

def ex4a():
    # (4a) Creates a new DataFrame by joining together two existing DataFrames
    # based on a common data column, in this case matching on department_id
    print('4a')
    joined = employees.join(
    departments,
    on="department_id",
    how="inner"
    )
    joined.show()

def ex4b():
    # (4b) join can also combine datasets based on conditional logic, 
    # which for example is helpful if DataFrames dont have the same header.

    print('4b')
    departments2 = departments.withColumnRenamed(
    "department_id",
    "id"
    )
    
    joined = employees.join(
    departments2,
    employees.department_id == departments2.id,
    how="inner"
    ).show()

def test_all():
    ex1_1a()
    ex1_1b()
    ex1_1c()
    ex1_2a()
    ex1_2b()
    ex1_2c()
    ex3()
    ex4a()
    ex4b()

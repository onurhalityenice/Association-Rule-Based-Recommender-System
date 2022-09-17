##############################################################
# ASSOCIATION RULE BASED RECOMMENDER SYSTEM
##############################################################

# Variables:
# UserId: Customer number
# ServiceId: Anonymized services belonging to each category. (Example: Upholstery washing service under the cleaning category)
# A ServiceId can be found under different categories and refers to different services under different categories.
# Example: While the service with CategoryId 7 and ServiceId 4 is heating cleaning, the service with CategoryId 2 and ServiceId 4 is furniture assembly
# CategoryId: Anonymized categories. (Example: Cleaning, transportation, renovation category)
# CreateDate: The date the service was purchased

#####################################################
# Part 1:  Preparing The Data
#####################################################

# Reading the armut_data.csv file
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("Miuul/WEEK_5/datasets/armut_data.csv")
df.head()
df.describe().T
df.isnull().sum()

# ServiceID represents a different service for each CategoryID.
# Crearing a new variable to represent these services by combining ServiceID and CategoryID with "_".
df["Service"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()

# The data set consists of the date and time the services are received, there is no basket definition (invoice, etc.).
# In order to apply Association Rule Learning, a basket (invoice, etc.) definition must be created.
# Basket definition is the services that each customer receives monthly.
# Baskets must be identified with a unique ID.
# For this, first a new date variable will be created that contains only the year and month.
# Combining UserID and newly created date variable with "_" and assign a new variable named ID.
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["New_Date"] = df["CreateDate"].dt.strftime('%Y-%m')

df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()

#####################################################
# Part 2:  Generating Association Rules
#####################################################

# Creating a basket-service-pivot table.
df.pivot_table(columns=["Service"], index=["SepetID"], values=["Service"], aggfunc="count").head()

invoice_product_df = df.pivot_table(columns=["Service"],
                                    index=["SepetID"],
                                    values=["Service"],
                                    aggfunc="count").fillna(0).applymap(lambda x: 1 if x > 0 else 0)

#Creating association rules
frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    return recommendation_list[:rec_count]

# Using the arl_recommender function to recommend a service to a user who has received the 2_0 service in the last 1 month
arl_recommender(rules, "2_0", 5)

### THE END ###
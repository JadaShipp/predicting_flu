


# Determine if a string is a palendrome 

import regex as re

def is_palendrome(some_words) --> bool:
    clean_words = (re.sub[0-9], "").lower()

    reversed_string = ""

    for x in range (1, len(clean_words +1)):
    
        if clean_words[x] == clean_words[-x]
        continue
        else 
        return False

Timeit 



assert tacocat == True
assert c00lkidz == False 
assert taco-cat == True
asserrt thisisalongstring == False



Table BASE
PROD_ID, 
STORE_NBR, BASE_PRICE
1, 55, .50
1. 12, .40
2, 66, .70
3, 44, 1.00
3, 12, .30
4, 12, .10
12, 12, .5
***********************************************

Table STORE_EXCP
PROD_ID, 
STR_NBR, 
STORE_PRICE
1, 55, .40
***********************************************
TABLE AD
PROD_ID, 
STR_NBR, 
AD_PRICE
2, 66, .40
***********************************************
TABLE EXCEPTION
PROD_ID, STR_NBR, EXCEPTION_PRICE
1, 55, .30
***********************************************

Base table will allways have a record for a product
EXCEPTION --> AD --> STORE --> BASE
***********************************************
Write a query which will give the current price of all product and all stores
PROD_ID, 
STORE_NBR, 
SALE_PRICE
1, 55, .30
2, 66, .40
3, 44. 1.00
3, 12, .30

Select base_tabel.prod_id, 
base_tabel.store_nbr, 
base_table.SALE_PRICE, 
table_exception.exception_price,
add_price,
store_price
From 
Base_table
Left JOIN table_exception on prod_id
Left JOIN a on prod_id and store_nbr
Left Join b on prod_id and STORE_NBR
Left Join c on prod_id and Store_nbr



***********************************************


Write a query to find the store where product 2 is sold at a cheapest price.
PROD_ID, 
STORE_NBR, 
SALE_PRICE
2, 66, .40

Select
MIN(sale_price, store_price, ad_price, exception_price)
WHERE prod_id = 2 
***********************************************
Write a query to find the count of products carried by store each store
STORE_NBR, 
PROD_COUNT
55, 1
12, 4
66, 1
44, 1

Select store_nbr, Count(prod_id)
From Base table
Groupby store_nbr
---------------------------------------------------------------------------
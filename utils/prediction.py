import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


def make_prediction(model):

    purchase_amount = float(input("Purchase Amount ............. "))
    engagement_score = float(input("Engagement Score ............ "))
    purchase_interest_score = float(input("Purchase Interest Score ..... "))
    customer_rating = float(input("Customer Rating ............. "))
    purchase_date = input("Purchase Date (YYYY-MM-DD)... ")

    print("\nSocial Media Platform")
    print("1. Facebook")
    print("2. Instagram")
    print("3. LinkedIn")
    print("4. TikTok")
    print("5. Twitter")

    while True:

        choice = input("\nChoice: ")

        if choice == "1":
            social_media_platform = "Facebook"
            break

        elif choice == "2":
            social_media_platform = "Instagram"
            break

        elif choice == "3":
            social_media_platform = "LinkedIn"
            break

        elif choice == "4":
            social_media_platform = "TikTok"
            break

        elif choice == "5":
            social_media_platform = "Twitter"
            break

        else:
            print("Invalid choice. Please try again.")

    print("\nReview Sentiment")
    print("1. Negative")
    print("2. Neutral")
    print("3. Positive")

    while True:

        choice = input("\nChoice: ")

        if choice == "1":
            review_sentiment = "Negative"
            break

        elif choice == "2":
            review_sentiment = "Neutral"
            break

        elif choice == "3":
            review_sentiment = "Positive"
            break

        else:
            print("Invalid choice. Please try again.")

    purchase_date = datetime.strptime(purchase_date, "%Y-%m-%d")
    month = purchase_date.month
    day = purchase_date.day
    day_of_week = purchase_date.weekday()
    is_weekend = int(day_of_week >= 5)

    X = pd.DataFrame({

        "engagement_score": [engagement_score],
        "purchase_interest_score": [purchase_interest_score],
        "purchase_amount": [purchase_amount],
        "customer_rating": [customer_rating],
        "month": [month],
        "day": [day],
        "day_of_week": [day_of_week],
        "is_weekend": [is_weekend],
        "social_media_platform_Instagram": [int(social_media_platform == "Instagram")],
        "social_media_platform_LinkedIn": [int(social_media_platform == "LinkedIn")],
        "social_media_platform_TikTok": [int(social_media_platform == "TikTok")],
        "social_media_platform_Twitter": [int(social_media_platform == "Twitter")],
        "review_sentiment_Neutral": [int(review_sentiment == "Neutral")],
        "review_sentiment_Positive": [int(review_sentiment == "Positive")]
    })

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).max()

    #le = LabelEncoder()
    #recommended_product = le.inverse_transform([prediction])[0]

    print("\n" + "=" * 55)
    print("  ------- Prediction Result -------")
    
    #print(f"Recommended Product : {recommended_product}")
    print(f"Recommended Product : {prediction}")
    #print(f"Confidence          : {probability:.2%}")
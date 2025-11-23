# ============================================================================
# დიაბეტის პროგნოზირების კლასიფიკაცია
# ============================================================================
# ეს სკრიპტი აჩვენებს კლასიფიკაციას pipeline-ებითა და ჰიპერპარამეტრების დარეგულირებით:
# 1. Naive Bayes MinMaxScaler-ით Pipeline-ში
# 2. Random Forest GridSearchCV-ით ჰიპერპარამეტრების ოპტიმიზაციისთვის
# მონაცემთა ნაკრები: diabetes.csv (Pima Indians Diabetes Database)
# ============================================================================

# საჭირო ბიბლიოთეკების შემოტანა
import pandas as pd  # მონაცემთა მანიპულაციისთვის
from sklearn.naive_bayes import GaussianNB  # Naive Bayes კლასიფიკატორი
from sklearn.ensemble import RandomForestClassifier  # Random Forest კლასიფიკატორი
from sklearn.preprocessing import MinMaxScaler  # ფუნქციების [0,1] დიაპაზონში მასშტაბირებისთვის
from sklearn.pipeline import Pipeline  # წინასწარი დამუშავებისა და მოდელის ერთად დაკავშირებისთვის
from sklearn.model_selection import train_test_split, GridSearchCV  # მონაცემების დაყოფისა და ჰიპერპარამეტრების დარეგულირებისთვის

# ============================================================================
# მონაცემების ჩატვირთვა და გამოკვლევა
# ============================================================================

# დიაბეტის მონაცემთა ნაკრების ჩატვირთვა
data = pd.read_csv("diabetes.csv")

print("=" * 70)
print("დიაბეტის მონაცემთა ნაკრები")
print("=" * 70)
print("მონაცემთა ნაკრების პირველი 5 მწკრივი:")
print(data.head())
print(f"\nმონაცემთა ნაკრების ფორმა: {data.shape}")
print(f"\nსვეტების სახელები: {list(data.columns)}")
print(f"\nსამიზნე ცვლადის (Outcome) მნიშვნელობები: {data['Outcome'].unique()}")
print("  0 = დიაბეტი არ არის, 1 = დიაბეტი")
print("\n")

# ============================================================================
# მონაცემების მომზადება
# ============================================================================

# პროგნოზირებისთვის ფუნქციების (დამოუკიდებელი ცვლადების) შერჩევა
# ეს არის სამედიცინო გაზომვები, რომლებიც შეიძლება მიუთითებდეს დიაბეტის რისკზე
X = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].values

# სამიზნე ცვლადის (დამოკიდებული ცვლადი) შერჩევა - რისი პროგნოზირებაც გვინდა
y = data['Outcome'].values

# მონაცემების სწავლების (70%) და ტესტირების (30%) სეტებად დაყოფა
# random_state=1 უზრუნველყოფს განმეორებადობას
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print("=" * 70)
print("მონაცემების დაყოფა")
print("=" * 70)
print(f"სწავლების სეტის ზომა: {X_train.shape[0]} ნიმუში")
print(f"ტესტირების სეტის ზომა: {X_test.shape[0]} ნიმუში")
print("\n")

# ============================================================================
# მოდელი 1: NAIVE BAYES PIPELINE-ით
# ============================================================================

print("=" * 70)
print("NAIVE BAYES კლასიფიკატორი PIPELINE-ით")
print("=" * 70)

# Pipeline-ის შექმნა, რომელიც აერთიანებს წინასწარ დამუშავებასა და მოდელს
# ნაბიჯი 1: MinMaxScaler - ყველა ფუნქციას მასშტაბირებს [0, 1] დიაპაზონში
# ნაბიჯი 2: GaussianNB - იყენებს Naive Bayes კლასიფიკაციას
# Pipeline უზრუნველყოფს, რომ scaler მხოლოდ სწავლების მონაცემებზე იყოს მორგებული, რაც ხელს უშლის მონაცემების გაჟონვას
hybrid_nb = Pipeline(steps=[
    ("Scaler", MinMaxScaler()),  # წინასწარი დამუშავების ნაბიჯი
    ("algo1", GaussianNB())      # მოდელის ნაბიჯი
])

# Pipeline-ის მორგება სწავლების მონაცემებზე
# ეს ამორგებს როგორც scaler-ს, ასევე მოდელს
hybrid_nb.fit(X_train, y_train)

# ტესტის მონაცემებზე სიზუსტის გამოთვლა
# Pipeline ავტომატურად იყენებს მასშტაბირებას პროგნოზამდე
accuracy_nb = hybrid_nb.score(X_test, y_test)

print(f"Naive Bayes სიზუსტე: {accuracy_nb:.4f}")
print(f"Naive Bayes სიზუსტე: {accuracy_nb*100:.2f}%")
print("\n")

# ============================================================================
# მოდელი 2: RANDOM FOREST დარეგულირების გარეშე (ბაზისური ხაზი)
# ============================================================================

print("=" * 70)
print("RANDOM FOREST კლასიფიკატორი (ბაზისური ხაზი)")
print("=" * 70)

# Random Forest მოდელის შექმნა ხელით პარამეტრებით
# n_estimators=100 ნიშნავს 100 გადაწყვეტილების ხეს ტყეში
# max_depth=2 ზღუდავს ხის სიღრმეს გადაჭარბებული მორგების თავიდან ასაცილებლად
model_rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=2)

# მოდელის სწავლება
model_rf_baseline.fit(X_train, y_train)

# სიზუსტის გამოთვლა
accuracy_rf_baseline = model_rf_baseline.score(X_test, y_test)

print(f"Random Forest (ბაზისური ხაზი) სიზუსტე: {accuracy_rf_baseline:.4f}")
print(f"Random Forest (ბაზისური ხაზი) სიზუსტე: {accuracy_rf_baseline*100:.2f}%")
print("\n")

# ============================================================================
# მოდელი 3: RANDOM FOREST GRID SEARCH-ით (ოპტიმიზებული)
# ============================================================================

print("=" * 70)
print("RANDOM FOREST GRID SEARCH ოპტიმიზაციით")
print("=" * 70)

# Random Forest მოდელის შექმნა (პარამეტრები დარეგულირდება)
model_rf = RandomForestClassifier()

# ძიებისთვის პარამეტრების ბადის განსაზღვრა
# GridSearchCV შეამოწმებს ამ პარამეტრების ყველა კომბინაციას
parameters = dict()
parameters['n_estimators'] = [20, 30, 10, 50]  # ხეების რაოდენობა, რომელიც უნდა შემოწმდეს
parameters['max_depth'] = [2, 3, 4, 5, 6]      # ხეების მაქსიმალური სიღრმე, რომელიც უნდა შემოწმდეს

print("პარამეტრების ბადე:")
print(f"  n_estimators (ხეების რაოდენობა): {parameters['n_estimators']}")
print(f"  max_depth (ხის სიღრმე): {parameters['max_depth']}")
print(f"  შემოწმებადი კომბინაციების საერთო რაოდენობა: {len(parameters['n_estimators']) * len(parameters['max_depth'])}")
print("\nსაუკეთესო პარამეტრების ძიება...")

# GridSearchCV ობიექტის შექმნა
# estimator: ოპტიმიზაციისთვის მოდელი
# param_grid: შესამოწმებელი პარამეტრები
# scoring: ოპტიმიზაციის მეტრიკა (სიზუსტე)
# n_jobs=-1: პარალელური დამუშავებისთვის ყველა CPU ბირთვის გამოყენება
# cv=5: 5-ჯერადი ჯვარედინი ვალიდაციის გამოყენება
hybrid_rf = GridSearchCV(
    estimator=model_rf,
    param_grid=parameters,
    scoring='accuracy',
    n_jobs=-1,
    cv=5  # 5-ჯერადი ჯვარედინი ვალიდაცია
)

# GridSearchCV-ის მორგება - ეს ასწავლის მოდელებს ყველა პარამეტრის კომბინაციით
hybrid_rf.fit(X_train, y_train)

# საუკეთესო პარამეტრებით ნაპოვნი სიზუსტის მიღება
accuracy_rf_optimized = hybrid_rf.score(X_test, y_test)

# GridSearch-ის მიერ ნაპოვნი საუკეთესო პარამეტრების მიღება
best_params = hybrid_rf.best_params_

print("\nბადის ძიება დასრულებულია!")
print(f"Random Forest (ოპტიმიზებული) სიზუსტე: {accuracy_rf_optimized:.4f}")
print(f"Random Forest (ოპტიმიზებული) სიზუსტე: {accuracy_rf_optimized*100:.2f}%")
print(f"\nნაპოვნი საუკეთესო პარამეტრები:")
print(f"  n_estimators (ხეების რაოდენობა): {best_params['n_estimators']}")
print(f"  max_depth (ხის სიღრმე): {best_params['max_depth']}")
print("\n")

# ============================================================================
# დეტალური GRID SEARCH შედეგები
# ============================================================================

print("=" * 70)
print("დეტალური GRID SEARCH შედეგები")
print("=" * 70)

# GridSearch შედეგების DataFrame-ად გადაქცევა მარტივი ნახვისთვის
results_df = pd.DataFrame(hybrid_rf.cv_results_)

# შესაბამისი სვეტების შერჩევა და ჩვენება
print("\nსაშუალო ტესტის ქულით ტოპ 5 პარამეტრის კომბინაცია:")
print(results_df[['param_n_estimators', 'param_max_depth', 'mean_test_score', 'rank_test_score']]
      .sort_values('rank_test_score')
      .head(5)
      .to_string(index=False))
print("\n")

# ============================================================================
# მოდელების შედარება
# ============================================================================

print("=" * 70)
print("მოდელების შედარების შეჯამება")
print("=" * 70)
print(f"Naive Bayes სიზუსტე:              {accuracy_nb:.4f} ({accuracy_nb*100:.2f}%)")
print(f"Random Forest (ბაზისური ხაზი) სიზუსტე: {accuracy_rf_baseline:.4f} ({accuracy_rf_baseline*100:.2f}%)")
print(f"Random Forest (ოპტიმიზებული) სიზუსტე: {accuracy_rf_optimized:.4f} ({accuracy_rf_optimized*100:.2f}%)")
print(f"\nბაზისური ხაზიდან ოპტიმიზებულზე გაუმჯობესება: {(accuracy_rf_optimized - accuracy_rf_baseline)*100:.2f}%")
print("=" * 70)

# ============================================================================
# ძირითადი შეხედულებები
# ============================================================================

print("\n" + "=" * 70)
print("ძირითადი შეხედულებები")
print("=" * 70)
print("1. Pipeline-ის სარგებელი:")
print("   - ხელს უშლის მონაცემების გაჟონვას scaler-ის მხოლოდ სწავლების მონაცემებზე მორგებით")
print("   - ამარტივებს კოდს წინასწარი დამუშავებისა და მოდელირების ერთად დაკავშირებით")
print("   - ამარტივებს განლაგებას")
print("\n2. GridSearchCV-ის სარგებელი:")
print("   - ავტომატურად პოულობს საუკეთესო ჰიპერპარამეტრებს")
print("   - იყენებს ჯვარედინ ვალიდაციას გადაჭარბებული მორგების თავიდან ასაცილებლად")
print("   - დაზოგავს დროს ხელით დარეგულირებასთან შედარებით")
print("\n3. მოდელის შერჩევა:")
if accuracy_nb > accuracy_rf_optimized:
    print("   - Naive Bayes ამ მონაცემთა ნაკრებისთვის საუკეთესო შედეგი აჩვენა")
    print("   - Naive Bayes უფრო მარტივი და სწრაფია")
elif accuracy_rf_optimized > accuracy_nb:
    print("   - ოპტიმიზებულმა Random Forest-მა საუკეთესო შედეგი აჩვენა")
    print("   - Random Forest შეუძლია რთული ნიმუშების დაჭერა")
else:
    print("   - ორივე მოდელმა თანაბრად კარგი შედეგი აჩვენა")
print("=" * 70)

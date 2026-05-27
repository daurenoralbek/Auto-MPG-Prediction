# Qadam Mentor — Колледж Ментор-Іздеу Қосымшасы

> **Студент:** Оралбек Каусар Ин  
> **ID:** 080626652754  
> **Технология:** Flutter + Firebase  

---

## Қосымша туралы

**Qadam Mentor** — колледж студенттерін мұғалімдер, ағалық студенттер және клуб жетекшілерімен байланыстыратын мобильді қосымша.

### Шешілетін мәселе
Студенттер қандай мұғалімге жүгінерін, IELTS-ке қалай дайындалатынын, шетелде оқуды қалай ұйымдастыратынын білмейді. Қосымша бұл кедергіні жояды.

---

## Мүмкіндіктер

### Аутентификация
- Email/пароль арқылы тіркелу және кіру
- Google Sign-In
- Пароль қалпына келтіру
- Рөл таңдау (студент / ментор)

### Менторлар
- Менторлар тізімін көру
- Санат бойынша сүзгілеу (IELTS, шетелде оқу, бухгалтерия және т.б.)
- Іздеу
- Ментор профилін толық қарау
- Рейтинг және пікірлер
- Қол жетімділік кестесі

### Брондау
- Кеңес брондау (күн + уақыт + тақырып)
- Брондау тарихы (күтуде / белсенді / аяқталған)
- Брондауды жою

### Чат
- Менторлармен нақты уақытта хат алмасу
- Хабарлар тізімі

### Профиль
- Профиль суретін жүктеу (камера / галерея)
- Профильді өңдеу
- Статистика

### Таңдаулылар
- Менторларды таңдаулыларға қосу / алып тастау

---

## Экрандар (25+)

| № | Экран | Сипаттама |
|---|-------|-----------|
| 1 | Splash | Логотип анимациясы |
| 2 | Onboarding 1 | Ментор табыңыз |
| 3 | Onboarding 2 | Тікелей байланысыңыз |
| 4 | Onboarding 3 | Мақсатыңызға жетіңіз |
| 5 | Кіру | Email/Google |
| 6 | Тіркелу | Email + рөл |
| 7 | Пароль қалпына | Сілтеме жіберу |
| 8 | Бас бет | Санаттар + үздік менторлар |
| 9 | Менторлар | Тізім + іздеу + сүзгі |
| 10 | Ментор профилі | Толық мәлімет |
| 11 | Санаттар | Барлық санаттар |
| 12 | Іздеу | Нақты уақытта іздеу |
| 13 | Сүзгі | Санат + рөл + қол жетімділік |
| 14 | Хабарлар тізімі | Барлық сөйлесулер |
| 15 | Чат | Нақты уақытта хат алмасу |
| 16 | Брондау | Күн + уақыт + тақырып |
| 17 | Брондау тарихы | 3 қойынды |
| 18 | Сұрау жіберілді | Растау экраны |
| 19 | Таңдаулылар | Сақталған менторлар |
| 20 | Профиль | Пайдаланушы профилі |
| 21 | Профильді өңдеу | Сурет + мәлімет |
| 22 | Хабарландырулар | Жүйе хабарлары |
| 23 | Параметрлер | Баптаулар мәзірі |
| 24 | Қосымша туралы | Нұсқа + жасаушы |
| 25 | Анықтама | FAQ |

---

## Firebase Деректер Базасы

### Коллекциялар

```
/users/{uid}
  uid, email, name, role, photoUrl, bio, specialty, year, phone, createdAt

/mentors/{mentorId}
  uid, name, email, photoUrl, bio, specialty, categories[], rating,
  reviewCount, isAvailable, availableDays[], availableHours, role, createdAt

/bookings/{bookingId}
  studentId, mentorId, studentName, mentorName, date, timeSlot,
  topic, message, status, createdAt

/chats/{chatId}
  participants[], studentId, mentorId, studentName, mentorName,
  lastMessage, lastMessageTime, unreadCount

/chats/{chatId}/messages/{msgId}
  senderId, text, timestamp, isRead

/favorites/{userId_mentorId}
  userId, mentorId, createdAt
```

---

## Орнату

### 1. Талаптар
- Flutter 3.19+ орнатылған болуы керек
- Firebase жобасы жасалған болуы керек

### 2. Жобаны клондау
```bash
git clone <repo-url>
cd qadam_mentor
flutter pub get
```

### 3. Firebase баптау
1. [Firebase Console](https://console.firebase.google.com) → Жаңа жоба жасаңыз
2. Android қосымшасын қосыңыз: `com.example.qadam_mentor`
3. `google-services.json` → `android/app/` қойыңыз
4. FlutterFire CLI орнатыңыз:
   ```bash
   dart pub global activate flutterfire_cli
   flutterfire configure
   ```
5. Firebase Authentication → Email/Password + Google қосыңыз
6. Firestore Database → Test mode-пен жасаңыз
7. Firebase Storage → Қосыңыз

### 4. Іске қосу
```bash
flutter run
```

### 5. APK жасау
```bash
flutter build apk --release
# APK: build/app/outputs/flutter-apk/app-release.apk
```

---

## Сынама деректер

Firestore-ға сынама ментор қосу:
```json
Collection: mentors
Document ID: test_mentor_1
{
  "uid": "test_mentor_1",
  "name": "Айжан Нұрмаханова",
  "email": "aigerim@test.com",
  "bio": "5 жылдық IELTS дайындығы тәжірибесі бар мұғалім",
  "specialty": "Ағылшын тілі / IELTS",
  "categories": ["ielts"],
  "rating": 4.8,
  "reviewCount": 24,
  "isAvailable": true,
  "availableDays": ["Дүйсенбі", "Сейсенбі", "Сәрсенбі"],
  "availableHours": "14:00–18:00",
  "role": "teacher",
  "createdAt": <Timestamp>
}
```

---

## Презентация стратегиясы

### Демо CRUD ағымы
1. **Create** — Жаңа тіркелгі жасаңыз (Register)
2. **Read** — Менторлар тізімін қараңыз
3. **Create** — Брондау сұрауын жіберіңіз
4. **Update** — Профильді өңдеңіз (сурет жүктеңіз)
5. **Delete** — Брондауды жойыңыз
6. **Realtime** — Екінші аккаунттан чат хабарын жіберіңіз

### Тексерулер
- ✅ Firebase Console → Authentication → Users
- ✅ Firebase Console → Firestore → users / mentors / bookings
- ✅ Firebase Console → Storage → profile_images

### Ескертулер
- Демо алдында сынама менторларды Firestore-ға қолмен қосыңыз
- Интернет байланысын тексеріңіз
- Брондау жіберілгенін Firestore-да тікелей көрсетіңіз

---

## Жоба құрылымы

```
lib/
├── main.dart              # Бастапқы нүкте
├── app.dart               # MaterialApp + маршруттар
├── firebase_options.dart  # Firebase конфигурациясы
├── config/                # Тақырып, түстер, тұрақтылар
├── models/                # Деректер модельдері
├── services/              # Firebase сервистері
├── providers/             # Күй басқару (Provider)
├── screens/               # 25+ экран
└── widgets/               # Ортақ виджеттер
```

---

*Qadam Mentor © 2024 — Оралбек Каусар Ин*

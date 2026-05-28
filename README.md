# JIHC TeamUp

**Student:** Kausar Oralbek | **ID:** 080626652754

A full-featured Flutter team collaboration app built for JIHC students.

---

## Features

- Google Sign-In + Email/Password Auth
- Cloud Firestore real-time CRUD
- Firebase Storage image upload
- Real-time chat (StreamBuilder)
- 25+ screens with modern blue/white UI
- Bottom navigation (Home, Teams, Chat, Alerts, Profile)
- Join request system with accept/reject
- Portfolio showcase
- Category-based post filtering
- Search posts
- Onboarding flow

---

## pubspec.yaml Dependencies

```yaml
firebase_core: ^3.1.0
firebase_auth: ^5.1.0
cloud_firestore: ^5.1.0
firebase_storage: ^12.1.0
google_sign_in: ^6.2.1
cached_network_image: ^3.3.1
image_picker: ^1.1.2
shimmer: ^3.0.0
timeago: ^3.6.1
intl: ^0.19.0
shared_preferences: ^2.2.3
```

---

## Firebase Setup

1. Create a Firebase project at https://console.firebase.google.com
2. Enable **Authentication** → Email/Password + Google
3. Create **Firestore Database** in production mode
4. Enable **Firebase Storage**
5. Download `google-services.json` → place in `android/app/`
6. Replace placeholder values in `kFirebaseOptions` inside `main.dart`

---

## Firestore Collections

| Collection    | Purpose                      |
|---------------|------------------------------|
| `/users`      | User profiles                |
| `/team_posts` | Team recruitment posts       |
| `/messages`   | Direct messages              |
| `/requests`   | Join requests                |

---

## Firestore Security Rules

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {

    match /users/{userId} {
      allow read: if request.auth != null;
      allow write: if request.auth != null && request.auth.uid == userId;
    }

    match /team_posts/{postId} {
      allow read: if request.auth != null;
      allow create: if request.auth != null;
      allow update, delete: if request.auth != null
        && request.auth.uid == resource.data.authorId;
    }

    match /messages/{msgId} {
      allow read: if request.auth != null
        && (request.auth.uid == resource.data.senderId
         || request.auth.uid == resource.data.receiverId);
      allow create: if request.auth != null;
    }

    match /requests/{reqId} {
      allow read: if request.auth != null
        && (request.auth.uid == resource.data.senderId
         || request.auth.uid == resource.data.ownerId);
      allow create: if request.auth != null;
      allow update: if request.auth != null
        && request.auth.uid == resource.data.ownerId;
    }
  }
}
```

---

## Sample Dummy Data (Firestore)

### /users/{uid}
```json
{
  "uid": "abc123",
  "name": "Kausar Oralbek",
  "email": "kausar@jihc.edu",
  "photoUrl": "",
  "bio": "Flutter developer & ML enthusiast",
  "skills": ["Flutter", "Python", "Firebase"],
  "major": "Software Engineering",
  "year": "3rd Year",
  "createdAt": "2025-01-01T00:00:00Z"
}
```

### /team_posts/{postId}
```json
{
  "title": "Need Flutter Dev for Hackathon",
  "description": "Looking for an experienced Flutter developer to build a mobile app for JIHC Hackathon 2025.",
  "category": "Hackathon",
  "teamSize": 3,
  "skills": ["Flutter", "Firebase", "UI/UX"],
  "deadline": "Dec 15, 2025",
  "authorId": "abc123",
  "authorName": "Kausar Oralbek",
  "authorPhoto": "",
  "likes": 5,
  "requestCount": 2,
  "createdAt": "2025-05-01T00:00:00Z"
}
```

### /messages/{msgId}
```json
{
  "chatId": "abc123_def456",
  "senderId": "abc123",
  "senderName": "Kausar Oralbek",
  "senderPhoto": "",
  "receiverId": "def456",
  "text": "Hey! I saw your post, I'm interested in joining.",
  "sentAt": "2025-05-01T10:30:00Z"
}
```

### /requests/{reqId}
```json
{
  "postId": "post123",
  "postTitle": "Need Flutter Dev for Hackathon",
  "senderId": "def456",
  "senderName": "Assel Nurova",
  "senderPhoto": "",
  "ownerId": "abc123",
  "status": "pending",
  "createdAt": "2025-05-02T09:00:00Z"
}
```

---

## Screens (25+)

1. Splash Screen
2. Onboarding 1 — Find Your Team
3. Onboarding 2 — Build Together
4. Onboarding 3 — Grow & Succeed
5. Login
6. Register
7. Forgot Password
8. Home (quick actions + recent posts)
9. Team Posts Feed (with category filter)
10. Post Details
11. Create Team Post
12. Edit Team Post
13. Categories Grid
14. Category Posts List
15. Search
16. Chat List
17. Chat Screen
18. Notifications
19. Join Requests
20. Accepted Teams
21. Profile
22. Edit Profile
23. Portfolio
24. Settings
25. About App
26. Help & Support

---

## Running the App

```bash
flutter pub get
flutter run
```

> Make sure `google-services.json` is in `android/app/` before running.

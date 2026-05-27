class AppConstants {
  AppConstants._();

  // Firestore collection names
  static const String usersCollection = 'users';
  static const String mentorsCollection = 'mentors';
  static const String bookingsCollection = 'bookings';
  static const String chatsCollection = 'chats';
  static const String messagesCollection = 'messages';
  static const String favoritesCollection = 'favorites';
  static const String notificationsCollection = 'notifications';

  // SharedPreferences keys
  static const String onboardingDoneKey = 'onboarding_done';

  // User roles
  static const String roleStudent = 'student';
  static const String roleMentor = 'mentor';

  // Booking statuses
  static const String statusPending = 'pending';
  static const String statusAccepted = 'accepted';
  static const String statusRejected = 'rejected';
  static const String statusCompleted = 'completed';

  // Mentor categories
  static const List<Map<String, String>> mentorCategories = [
    {'id': 'ielts', 'label': 'IELTS / Тіл', 'icon': '🌐'},
    {'id': 'abroad', 'label': 'Шетелде оқу', 'icon': '✈️'},
    {'id': 'accounting', 'label': 'Бухгалтерия', 'icon': '📊'},
    {'id': 'career', 'label': 'Мансап', 'icon': '🎯'},
    {'id': 'college_life', 'label': 'Колледж өмірі', 'icon': '🎓'},
    {'id': 'club', 'label': 'Клуб жетекшілері', 'icon': '🏆'},
    {'id': 'it', 'label': 'IT / Технология', 'icon': '💻'},
    {'id': 'science', 'label': 'Ғылым', 'icon': '🔬'},
  ];

  // Available time slots
  static const List<String> timeSlots = [
    '09:00',
    '10:00',
    '11:00',
    '12:00',
    '13:00',
    '14:00',
    '15:00',
    '16:00',
    '17:00',
  ];

  // Days of week in Kazakh
  static const List<String> weekDaysKk = [
    'Дүйсенбі',
    'Сейсенбі',
    'Сәрсенбі',
    'Бейсенбі',
    'Жұма',
    'Сенбі',
  ];
}

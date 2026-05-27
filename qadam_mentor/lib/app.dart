import 'package:flutter/material.dart';
import 'package:intl/date_symbol_data_local.dart';
import 'package:timeago/timeago.dart' as timeago;
import 'config/app_routes.dart';
import 'config/app_theme.dart';
import 'screens/splash/splash_screen.dart';
import 'screens/onboarding/onboarding_screen.dart';
import 'screens/auth/login_screen.dart';
import 'screens/auth/register_screen.dart';
import 'screens/auth/forgot_password_screen.dart';
import 'screens/main/main_screen.dart';
import 'screens/mentors/mentor_list_screen.dart';
import 'screens/mentors/mentor_detail_screen.dart';
import 'screens/mentors/categories_screen.dart';
import 'screens/mentors/search_screen.dart';
import 'screens/mentors/filter_screen.dart';
import 'screens/chat/chat_detail_screen.dart';
import 'screens/booking/booking_screen.dart';
import 'screens/booking/booking_history_screen.dart';
import 'screens/booking/request_submitted_screen.dart';
import 'screens/profile/edit_profile_screen.dart';
import 'screens/profile/mentor_profile_screen.dart';
import 'screens/notifications/notifications_screen.dart';
import 'screens/settings/settings_screen.dart';
import 'screens/settings/about_screen.dart';
import 'screens/settings/help_center_screen.dart';

class QadamMentorApp extends StatefulWidget {
  const QadamMentorApp({super.key});

  @override
  State<QadamMentorApp> createState() => _QadamMentorAppState();
}

class _QadamMentorAppState extends State<QadamMentorApp> {
  @override
  void initState() {
    super.initState();
    initializeDateFormatting('kk');
    timeago.setLocaleMessages('kk', _KkMessages());
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Qadam Mentor',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      initialRoute: AppRoutes.splash,
      routes: {
        AppRoutes.splash: (_) => const SplashScreen(),
        AppRoutes.onboarding: (_) => const OnboardingScreen(),
        AppRoutes.login: (_) => const LoginScreen(),
        AppRoutes.register: (_) => const RegisterScreen(),
        AppRoutes.forgotPassword: (_) => const ForgotPasswordScreen(),
        AppRoutes.main: (_) => const MainScreen(),
        AppRoutes.mentorList: (_) => const MentorListScreen(),
        AppRoutes.mentorDetail: (_) => const MentorDetailScreen(),
        AppRoutes.categories: (_) => const CategoriesScreen(),
        AppRoutes.search: (_) => const SearchScreen(),
        AppRoutes.filter: (_) => const FilterScreen(),
        AppRoutes.chatDetail: (_) => const ChatDetailScreen(),
        AppRoutes.booking: (_) => const BookingScreen(),
        AppRoutes.bookingHistory: (_) => const BookingHistoryScreen(),
        AppRoutes.requestSubmitted: (_) => const RequestSubmittedScreen(),
        AppRoutes.editProfile: (_) => const EditProfileScreen(),
        AppRoutes.mentorProfile: (_) => const MentorProfileScreen(),
        AppRoutes.notifications: (_) => const NotificationsScreen(),
        AppRoutes.settings: (_) => const SettingsScreen(),
        AppRoutes.about: (_) => const AboutScreen(),
        AppRoutes.helpCenter: (_) => const HelpCenterScreen(),
      },
    );
  }
}

class _KkMessages implements timeago.LookupMessages {
  @override
  String prefixAgo() => '';
  @override
  String prefixFromNow() => '';
  @override
  String suffixAgo() => 'бұрын';
  @override
  String suffixFromNow() => 'кейін';
  @override
  String lessThanOneMinute(int seconds) => 'жаңа ғана';
  @override
  String aboutAMinute(int minutes) => '1 минут';
  @override
  String minutes(int minutes) => '$minutes минут';
  @override
  String aboutAnHour(int minutes) => '1 сағат';
  @override
  String hours(int hours) => '$hours сағат';
  @override
  String aDay(int hours) => '1 күн';
  @override
  String days(int days) => '$days күн';
  @override
  String aboutAMonth(int days) => '1 ай';
  @override
  String months(int months) => '$months ай';
  @override
  String aboutAYear(int year) => '1 жыл';
  @override
  String years(int years) => '$years жыл';
  @override
  String wordSeparator() => ' ';
}

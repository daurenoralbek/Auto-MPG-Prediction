// ============================================================
// JIHC TeamUp — Full App in main.dart
// Student: Kausar Oralbek  |  ID: 080626652754
// ============================================================

import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:image_picker/image_picker.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:timeago/timeago.dart' as timeago;
import 'package:intl/intl.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:shimmer/shimmer.dart';

// ============================================================
// THEME CONSTANTS
// ============================================================
const Color kPrimary   = Color(0xFF1565C0);
const Color kPrimaryL  = Color(0xFF1E88E5);
const Color kAccent    = Color(0xFF42A5F5);
const Color kBg        = Color(0xFFF5F7FA);
const Color kCard      = Colors.white;
const Color kText      = Color(0xFF1A1A2E);
const Color kTextSub   = Color(0xFF6B7280);
const Color kDivider   = Color(0xFFE5E7EB);
const Color kSuccess   = Color(0xFF10B981);
const Color kError     = Color(0xFFEF4444);
const Color kWarning   = Color(0xFFF59E0B);

// ============================================================
// FIREBASE OPTIONS (replace with your real google-services values)
// ============================================================
const FirebaseOptions kFirebaseOptions = FirebaseOptions(
  apiKey:            'AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
  appId:             '1:000000000000:android:0000000000000000000000',
  messagingSenderId: '000000000000',
  projectId:         'jihc-teamup',
  storageBucket:     'jihc-teamup.appspot.com',
);

// ============================================================
// MAIN
// ============================================================
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.dark,
  ));
  try {
    await Firebase.initializeApp(options: kFirebaseOptions);
  } catch (_) {}
  runApp(const JihcTeamUpApp());
}

// ============================================================
// ROOT APP
// ============================================================
class JihcTeamUpApp extends StatelessWidget {
  const JihcTeamUpApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'JIHC TeamUp',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: kPrimary, brightness: Brightness.light),
        primaryColor: kPrimary,
        scaffoldBackgroundColor: kBg,
        fontFamily: 'Roboto',
        useMaterial3: true,
        appBarTheme: const AppBarTheme(
          backgroundColor: kPrimary,
          foregroundColor: Colors.white,
          elevation: 0,
          centerTitle: true,
          titleTextStyle: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: Colors.white),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: kPrimary,
            foregroundColor: Colors.white,
            minimumSize: const Size(double.infinity, 52),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: kDivider)),
          enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: kDivider)),
          focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: kPrimary, width: 2)),
          errorBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: kError)),
        ),
        cardTheme: CardTheme(
          color: kCard,
          elevation: 2,
          shadowColor: kPrimary.withOpacity(0.08),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        ),
      ),
      home: const SplashScreen(),
    );
  }
}

// ============================================================
// FIREBASE FUNCTIONS
// ============================================================
final _auth     = FirebaseAuth.instance;
final _firestore = FirebaseFirestore.instance;
final _storage  = FirebaseStorage.instance;
final _gSignIn  = GoogleSignIn();

Future<UserCredential?> signInWithGoogle() async {
  try {
    final googleUser = await _gSignIn.signIn();
    if (googleUser == null) return null;
    final googleAuth = await googleUser.authentication;
    final credential = GoogleAuthProvider.credential(
      accessToken: googleAuth.accessToken,
      idToken: googleAuth.idToken,
    );
    final uc = await _auth.signInWithCredential(credential);
    await _ensureUserDoc(uc.user!);
    return uc;
  } catch (e) {
    debugPrint('Google sign-in error: $e');
    return null;
  }
}

Future<UserCredential?> signInWithEmail(String email, String password) async {
  try {
    final uc = await _auth.signInWithEmailAndPassword(email: email, password: password);
    return uc;
  } catch (e) {
    debugPrint('Email sign-in error: $e');
    rethrow;
  }
}

Future<UserCredential?> registerWithEmail(String email, String password, String name) async {
  try {
    final uc = await _auth.createUserWithEmailAndPassword(email: email, password: password);
    await uc.user!.updateDisplayName(name);
    await _ensureUserDoc(uc.user!);
    return uc;
  } catch (e) {
    debugPrint('Register error: $e');
    rethrow;
  }
}

Future<void> _ensureUserDoc(User user) async {
  final ref = _firestore.collection('users').doc(user.uid);
  final snap = await ref.get();
  if (!snap.exists) {
    await ref.set({
      'uid': user.uid,
      'name': user.displayName ?? 'User',
      'email': user.email ?? '',
      'photoUrl': user.photoURL ?? '',
      'bio': '',
      'skills': <String>[],
      'major': '',
      'year': '',
      'createdAt': FieldValue.serverTimestamp(),
    });
  }
}

Future<void> signOut() async {
  await _gSignIn.signOut();
  await _auth.signOut();
}

Future<void> sendPasswordReset(String email) async {
  await _auth.sendPasswordResetEmail(email: email);
}

Future<String?> uploadImage(File file, String path) async {
  try {
    final ref = _storage.ref().child(path);
    final task = await ref.putFile(file);
    return await task.ref.getDownloadURL();
  } catch (e) {
    debugPrint('Upload error: $e');
    return null;
  }
}

Future<DocumentReference> createPost(Map<String, dynamic> data) async {
  return _firestore.collection('team_posts').add({
    ...data,
    'createdAt': FieldValue.serverTimestamp(),
    'authorId': _auth.currentUser!.uid,
    'authorName': _auth.currentUser!.displayName ?? 'User',
    'authorPhoto': _auth.currentUser!.photoURL ?? '',
    'likes': 0,
    'requestCount': 0,
  });
}

Future<void> updatePost(String id, Map<String, dynamic> data) async {
  await _firestore.collection('team_posts').doc(id).update({...data, 'updatedAt': FieldValue.serverTimestamp()});
}

Future<void> deletePost(String id) async {
  await _firestore.collection('team_posts').doc(id).delete();
}

Stream<QuerySnapshot> postsStream() {
  return _firestore.collection('team_posts').orderBy('createdAt', descending: true).snapshots();
}

Stream<QuerySnapshot> chatStream(String chatId) {
  return _firestore
      .collection('messages')
      .where('chatId', isEqualTo: chatId)
      .orderBy('sentAt', descending: true)
      .snapshots();
}

Future<void> sendMessage(String chatId, String text, String receiverId) async {
  final me = _auth.currentUser!;
  await _firestore.collection('messages').add({
    'chatId': chatId,
    'senderId': me.uid,
    'senderName': me.displayName ?? 'User',
    'senderPhoto': me.photoURL ?? '',
    'receiverId': receiverId,
    'text': text,
    'sentAt': FieldValue.serverTimestamp(),
  });
}

Future<void> sendJoinRequest(String postId, String postTitle, String ownerId) async {
  final me = _auth.currentUser!;
  await _firestore.collection('requests').add({
    'postId': postId,
    'postTitle': postTitle,
    'senderId': me.uid,
    'senderName': me.displayName ?? 'User',
    'senderPhoto': me.photoURL ?? '',
    'ownerId': ownerId,
    'status': 'pending',
    'createdAt': FieldValue.serverTimestamp(),
  });
  await _firestore.collection('team_posts').doc(postId).update({'requestCount': FieldValue.increment(1)});
}

Future<void> updateRequest(String id, String status) async {
  await _firestore.collection('requests').doc(id).update({'status': status});
}

Stream<QuerySnapshot> myRequestsStream() {
  final uid = _auth.currentUser?.uid ?? '';
  return _firestore.collection('requests').where('ownerId', isEqualTo: uid).snapshots();
}

Stream<QuerySnapshot> acceptedTeamsStream() {
  final uid = _auth.currentUser?.uid ?? '';
  return _firestore
      .collection('requests')
      .where('senderId', isEqualTo: uid)
      .where('status', isEqualTo: 'accepted')
      .snapshots();
}

Future<DocumentSnapshot> getUserDoc(String uid) async {
  return _firestore.collection('users').doc(uid).get();
}

Future<void> updateUserDoc(String uid, Map<String, dynamic> data) async {
  await _firestore.collection('users').doc(uid).update(data);
}

// ============================================================
// REUSABLE WIDGETS
// ============================================================

/// LOADING OVERLAY
Widget buildLoading() => const Center(
  child: CircularProgressIndicator(color: kPrimary),
);

/// EMPTY STATE
Widget buildEmpty(String msg, {IconData icon = Icons.inbox_outlined}) => Center(
  child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
    Icon(icon, size: 64, color: kTextSub),
    const SizedBox(height: 12),
    Text(msg, style: const TextStyle(color: kTextSub, fontSize: 15)),
  ]),
);

/// ERROR STATE
Widget buildError(String msg) => Center(
  child: Text(msg, style: const TextStyle(color: kError)),
);

/// AVATAR
Widget buildAvatar(String? url, {double r = 24}) => CircleAvatar(
  radius: r,
  backgroundColor: kPrimaryL,
  backgroundImage: (url != null && url.isNotEmpty) ? CachedNetworkImageProvider(url) : null,
  child: (url == null || url.isEmpty)
      ? Icon(Icons.person, color: Colors.white, size: r)
      : null,
);

/// CARD TILE
Widget buildCardTile({required Widget child, VoidCallback? onTap, EdgeInsets? padding}) => Card(
  margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
  child: InkWell(
    onTap: onTap,
    borderRadius: BorderRadius.circular(16),
    child: Padding(
      padding: padding ?? const EdgeInsets.all(16),
      child: child,
    ),
  ),
);

/// TAG CHIP
Widget buildTag(String label, {Color? bg}) => Container(
  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
  decoration: BoxDecoration(
    color: bg ?? kPrimary.withOpacity(0.1),
    borderRadius: BorderRadius.circular(20),
  ),
  child: Text(label, style: TextStyle(fontSize: 12, color: bg != null ? Colors.white : kPrimary, fontWeight: FontWeight.w600)),
);

/// SECTION HEADER
Widget buildSectionHeader(String title) => Padding(
  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
  child: Text(title, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: kText)),
);

/// DIVIDER
Widget kDividerWidget() => const Divider(color: kDivider, height: 1);

/// GRADIENT BUTTON
Widget buildGradientButton({required String label, required VoidCallback onTap, IconData? icon}) => GestureDetector(
  onTap: onTap,
  child: Container(
    height: 52,
    decoration: BoxDecoration(
      gradient: const LinearGradient(colors: [kPrimary, kPrimaryL]),
      borderRadius: BorderRadius.circular(14),
      boxShadow: [BoxShadow(color: kPrimary.withOpacity(0.3), blurRadius: 8, offset: const Offset(0, 4))],
    ),
    child: Center(
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        if (icon != null) ...[Icon(icon, color: Colors.white, size: 20), const SizedBox(width: 8)],
        Text(label, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700, fontSize: 16)),
      ]),
    ),
  ),
);

/// SHIMMER CARD
Widget buildShimmerCard() => Shimmer.fromColors(
  baseColor: Colors.grey[300]!,
  highlightColor: Colors.grey[100]!,
  child: Card(
    margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
    child: Container(height: 120),
  ),
);

/// SNACK
void showSnack(BuildContext ctx, String msg, {bool error = false}) {
  ScaffoldMessenger.of(ctx).showSnackBar(SnackBar(
    content: Text(msg),
    backgroundColor: error ? kError : kSuccess,
    behavior: SnackBarBehavior.floating,
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
  ));
}

// ============================================================
// SPLASH SCREEN
// ============================================================
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});
  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double> _fade;
  late Animation<double> _scale;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 1200));
    _fade  = CurvedAnimation(parent: _ctrl, curve: Curves.easeIn);
    _scale = Tween<double>(begin: 0.7, end: 1.0).animate(CurvedAnimation(parent: _ctrl, curve: Curves.elasticOut));
    _ctrl.forward();
    _navigate();
  }

  Future<void> _navigate() async {
    await Future.delayed(const Duration(seconds: 2));
    if (!mounted) return;
    final prefs = await SharedPreferences.getInstance();
    final seen  = prefs.getBool('onboarding_done') ?? false;
    if (_auth.currentUser != null) {
      Navigator.pushReplacement(context, _route(const MainShell()));
    } else if (!seen) {
      Navigator.pushReplacement(context, _route(const OnboardingScreen()));
    } else {
      Navigator.pushReplacement(context, _route(const LoginScreen()));
    }
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft, end: Alignment.bottomRight,
            colors: [kPrimary, kPrimaryL, kAccent],
          ),
        ),
        child: Center(
          child: FadeTransition(
            opacity: _fade,
            child: ScaleTransition(
              scale: _scale,
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(color: Colors.white.withOpacity(0.15), shape: BoxShape.circle),
                  child: const Icon(Icons.groups_2, size: 80, color: Colors.white),
                ),
                const SizedBox(height: 20),
                const Text('JIHC TeamUp', style: TextStyle(fontSize: 32, fontWeight: FontWeight.w900, color: Colors.white, letterSpacing: 1.5)),
                const SizedBox(height: 8),
                const Text('Connect · Collaborate · Create', style: TextStyle(fontSize: 14, color: Colors.white70, letterSpacing: 1)),
                const SizedBox(height: 40),
                const CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
                const SizedBox(height: 16),
                const Text('Student: Kausar Oralbek\nID: 080626652754',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 11, color: Colors.white54)),
              ]),
            ),
          ),
        ),
      ),
    );
  }
}

// ============================================================
// ONBOARDING SCREEN
// ============================================================
class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});
  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final _ctrl = PageController();
  int _page = 0;

  final _pages = const [
    _OnboardPage(
      icon: Icons.groups_2,
      title: 'Find Your Team',
      body: 'Discover talented teammates at JIHC. Browse projects and join teams that match your skills.',
      color: kPrimary,
    ),
    _OnboardPage(
      icon: Icons.rocket_launch_rounded,
      title: 'Build Together',
      body: 'Collaborate in real-time. Share ideas, delegate tasks and ship amazing projects together.',
      color: Color(0xFF0D47A1),
    ),
    _OnboardPage(
      icon: Icons.emoji_events_rounded,
      title: 'Grow & Succeed',
      body: 'Showcase your portfolio, win hackathons and build a reputation in the JIHC community.',
      color: kPrimaryL,
    ),
  ];

  void _done() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('onboarding_done', true);
    if (!mounted) return;
    Navigator.pushReplacement(context, _route(const LoginScreen()));
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(children: [
        PageView.builder(
          controller: _ctrl,
          itemCount: _pages.length,
          onPageChanged: (i) => setState(() => _page = i),
          itemBuilder: (_, i) => _pages[i],
        ),
        Positioned(
          bottom: 48,
          left: 24, right: 24,
          child: Column(children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(_pages.length, (i) => AnimatedContainer(
                duration: const Duration(milliseconds: 300),
                margin: const EdgeInsets.symmetric(horizontal: 4),
                width: _page == i ? 24 : 8,
                height: 8,
                decoration: BoxDecoration(
                  color: _page == i ? Colors.white : Colors.white38,
                  borderRadius: BorderRadius.circular(4),
                ),
              )),
            ),
            const SizedBox(height: 24),
            if (_page == _pages.length - 1)
              buildGradientButton(label: 'Get Started', onTap: _done, icon: Icons.arrow_forward)
            else
              Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
                TextButton(
                  onPressed: _done,
                  child: const Text('Skip', style: TextStyle(color: Colors.white70, fontSize: 15)),
                ),
                ElevatedButton(
                  onPressed: () => _ctrl.nextPage(duration: const Duration(milliseconds: 400), curve: Curves.easeInOut),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: kPrimary,
                    minimumSize: const Size(100, 46),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  child: const Text('Next'),
                ),
              ]),
          ]),
        ),
      ]),
    );
  }
}

class _OnboardPage extends StatelessWidget {
  final IconData icon;
  final String title, body;
  final Color color;
  const _OnboardPage({required this.icon, required this.title, required this.body, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter, end: Alignment.bottomCenter,
          colors: [color, color.withOpacity(0.7)],
        ),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Icon(icon, size: 120, color: Colors.white),
        const SizedBox(height: 32),
        Text(title, style: const TextStyle(fontSize: 28, fontWeight: FontWeight.w800, color: Colors.white)),
        const SizedBox(height: 16),
        Text(body, textAlign: TextAlign.center, style: const TextStyle(fontSize: 16, color: Colors.white80, height: 1.5)),
      ]),
    );
  }
}

// ============================================================
// LOGIN SCREEN
// ============================================================
class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});
  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _formKey  = GlobalKey<FormState>();
  final _emailCtrl = TextEditingController();
  final _passCtrl  = TextEditingController();
  bool _loading = false, _obscure = true;

  Future<void> _login() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _loading = true);
    try {
      await signInWithEmail(_emailCtrl.text.trim(), _passCtrl.text.trim());
      if (!mounted) return;
      Navigator.pushReplacement(context, _route(const MainShell()));
    } catch (e) {
      if (mounted) showSnack(context, 'Login failed: ${e.toString().split(']').last.trim()}', error: true);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _googleLogin() async {
    setState(() => _loading = true);
    final uc = await signInWithGoogle();
    if (!mounted) return;
    setState(() => _loading = false);
    if (uc != null) {
      Navigator.pushReplacement(context, _route(const MainShell()));
    } else {
      showSnack(context, 'Google sign-in cancelled.', error: true);
    }
  }

  @override
  void dispose() { _emailCtrl.dispose(); _passCtrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24),
          child: Form(
            key: _formKey,
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const SizedBox(height: 32),
              Center(child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(color: kPrimary.withOpacity(0.1), shape: BoxShape.circle),
                child: const Icon(Icons.groups_2, size: 56, color: kPrimary),
              )),
              const SizedBox(height: 16),
              const Center(child: Text('JIHC TeamUp', style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: kText))),
              const Center(child: Text('Sign in to continue', style: TextStyle(color: kTextSub, fontSize: 14))),
              const SizedBox(height: 36),
              const Text('Email', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 13, color: kText)),
              const SizedBox(height: 6),
              TextFormField(
                controller: _emailCtrl,
                keyboardType: TextInputType.emailAddress,
                decoration: const InputDecoration(prefixIcon: Icon(Icons.email_outlined, color: kPrimary), hintText: 'your@email.com'),
                validator: (v) => (v == null || !v.contains('@')) ? 'Enter valid email' : null,
              ),
              const SizedBox(height: 16),
              const Text('Password', style: TextStyle(fontWeight: FontWeight.w600, fontSize: 13, color: kText)),
              const SizedBox(height: 6),
              TextFormField(
                controller: _passCtrl,
                obscureText: _obscure,
                decoration: InputDecoration(
                  prefixIcon: const Icon(Icons.lock_outlined, color: kPrimary),
                  hintText: '••••••••',
                  suffixIcon: IconButton(
                    icon: Icon(_obscure ? Icons.visibility_off : Icons.visibility, color: kTextSub),
                    onPressed: () => setState(() => _obscure = !_obscure),
                  ),
                ),
                validator: (v) => (v == null || v.length < 6) ? 'Min 6 characters' : null,
              ),
              const SizedBox(height: 8),
              Align(
                alignment: Alignment.centerRight,
                child: TextButton(
                  onPressed: () => Navigator.push(context, _route(const ForgotPasswordScreen())),
                  child: const Text('Forgot Password?', style: TextStyle(color: kPrimary)),
                ),
              ),
              const SizedBox(height: 8),
              _loading
                  ? buildLoading()
                  : buildGradientButton(label: 'Sign In', onTap: _login, icon: Icons.login),
              const SizedBox(height: 16),
              Row(children: const [
                Expanded(child: Divider()), SizedBox(width: 12),
                Text('OR', style: TextStyle(color: kTextSub, fontSize: 12)),
                SizedBox(width: 12), Expanded(child: Divider()),
              ]),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: _loading ? null : _googleLogin,
                icon: const Icon(Icons.g_mobiledata, size: 28, color: Colors.red),
                label: const Text('Continue with Google', style: TextStyle(color: kText, fontWeight: FontWeight.w600)),
                style: OutlinedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 52),
                  side: const BorderSide(color: kDivider),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                ),
              ),
              const SizedBox(height: 24),
              Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                const Text("Don't have an account? ", style: TextStyle(color: kTextSub)),
                GestureDetector(
                  onTap: () => Navigator.push(context, _route(const RegisterScreen())),
                  child: const Text('Sign Up', style: TextStyle(color: kPrimary, fontWeight: FontWeight.w700)),
                ),
              ]),
            ]),
          ),
        ),
      ),
    );
  }
}

// ============================================================
// REGISTER SCREEN
// ============================================================
class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});
  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl  = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _passCtrl  = TextEditingController();
  final _cPassCtrl = TextEditingController();
  bool _loading = false, _obscure = true;

  Future<void> _register() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _loading = true);
    try {
      await registerWithEmail(_emailCtrl.text.trim(), _passCtrl.text.trim(), _nameCtrl.text.trim());
      if (!mounted) return;
      Navigator.pushAndRemoveUntil(context, _route(const MainShell()), (_) => false);
    } catch (e) {
      if (mounted) showSnack(context, 'Registration failed: ${e.toString().split(']').last.trim()}', error: true);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  void dispose() { _nameCtrl.dispose(); _emailCtrl.dispose(); _passCtrl.dispose(); _cPassCtrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Create Account')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Form(
          key: _formKey,
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const SizedBox(height: 8),
            _field('Full Name', _nameCtrl, Icons.person_outlined, validator: (v) => (v == null || v.isEmpty) ? 'Enter name' : null),
            const SizedBox(height: 14),
            _field('Email', _emailCtrl, Icons.email_outlined, type: TextInputType.emailAddress,
                validator: (v) => (v == null || !v.contains('@')) ? 'Enter valid email' : null),
            const SizedBox(height: 14),
            _field('Password', _passCtrl, Icons.lock_outlined, obscure: true,
                validator: (v) => (v == null || v.length < 6) ? 'Min 6 characters' : null),
            const SizedBox(height: 14),
            TextFormField(
              controller: _cPassCtrl,
              obscureText: _obscure,
              decoration: InputDecoration(
                labelText: 'Confirm Password',
                prefixIcon: const Icon(Icons.lock, color: kPrimary),
                suffixIcon: IconButton(
                  icon: Icon(_obscure ? Icons.visibility_off : Icons.visibility, color: kTextSub),
                  onPressed: () => setState(() => _obscure = !_obscure),
                ),
              ),
              validator: (v) => v != _passCtrl.text ? 'Passwords do not match' : null,
            ),
            const SizedBox(height: 24),
            _loading
                ? buildLoading()
                : buildGradientButton(label: 'Create Account', onTap: _register, icon: Icons.person_add),
            const SizedBox(height: 16),
            Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              const Text('Already have an account? ', style: TextStyle(color: kTextSub)),
              GestureDetector(
                onTap: () => Navigator.pop(context),
                child: const Text('Sign In', style: TextStyle(color: kPrimary, fontWeight: FontWeight.w700)),
              ),
            ]),
          ]),
        ),
      ),
    );
  }

  Widget _field(String label, TextEditingController ctrl, IconData icon, {
    bool obscure = false, TextInputType? type, FormFieldValidator<String>? validator
  }) => TextFormField(
    controller: ctrl,
    obscureText: obscure,
    keyboardType: type,
    decoration: InputDecoration(labelText: label, prefixIcon: Icon(icon, color: kPrimary)),
    validator: validator,
  );
}

// ============================================================
// FORGOT PASSWORD SCREEN
// ============================================================
class ForgotPasswordScreen extends StatefulWidget {
  const ForgotPasswordScreen({super.key});
  @override
  State<ForgotPasswordScreen> createState() => _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends State<ForgotPasswordScreen> {
  final _ctrl = TextEditingController();
  bool _loading = false, _sent = false;

  Future<void> _reset() async {
    if (_ctrl.text.trim().isEmpty) return;
    setState(() => _loading = true);
    try {
      await sendPasswordReset(_ctrl.text.trim());
      if (mounted) setState(() { _sent = true; _loading = false; });
    } catch (e) {
      if (mounted) { setState(() => _loading = false); showSnack(context, 'Error: $e', error: true); }
    }
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Reset Password')),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: _sent
            ? Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                const Icon(Icons.mark_email_read, size: 80, color: kSuccess),
                const SizedBox(height: 16),
                const Text('Email Sent!', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700)),
                const SizedBox(height: 8),
                Text('Check ${_ctrl.text} for password reset link.',
                    textAlign: TextAlign.center, style: const TextStyle(color: kTextSub)),
                const SizedBox(height: 24),
                ElevatedButton(onPressed: () => Navigator.pop(context), child: const Text('Back to Login')),
              ])
            : Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const SizedBox(height: 24),
                const Icon(Icons.lock_reset, size: 64, color: kPrimary),
                const SizedBox(height: 16),
                const Text('Forgot Password?', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700)),
                const SizedBox(height: 8),
                const Text('Enter your email to receive a reset link.', style: TextStyle(color: kTextSub)),
                const SizedBox(height: 24),
                TextFormField(
                  controller: _ctrl,
                  keyboardType: TextInputType.emailAddress,
                  decoration: const InputDecoration(labelText: 'Email', prefixIcon: Icon(Icons.email_outlined, color: kPrimary)),
                ),
                const SizedBox(height: 24),
                _loading ? buildLoading() : buildGradientButton(label: 'Send Reset Link', onTap: _reset, icon: Icons.send),
              ]),
      ),
    );
  }
}

// ============================================================
// MAIN SHELL (Bottom Navigation)
// ============================================================
class MainShell extends StatefulWidget {
  const MainShell({super.key});
  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _idx = 0;

  final _screens = const [
    HomeScreen(),
    TeamPostsFeedScreen(),
    ChatListScreen(),
    NotificationsScreen(),
    ProfileScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _idx, children: _screens),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _idx,
        onDestinationSelected: (i) => setState(() => _idx = i),
        backgroundColor: Colors.white,
        indicatorColor: kPrimary.withOpacity(0.12),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.home_outlined), selectedIcon: Icon(Icons.home, color: kPrimary), label: 'Home'),
          NavigationDestination(icon: Icon(Icons.groups_outlined), selectedIcon: Icon(Icons.groups, color: kPrimary), label: 'Teams'),
          NavigationDestination(icon: Icon(Icons.chat_outlined), selectedIcon: Icon(Icons.chat, color: kPrimary), label: 'Chat'),
          NavigationDestination(icon: Icon(Icons.notifications_outlined), selectedIcon: Icon(Icons.notifications, color: kPrimary), label: 'Alerts'),
          NavigationDestination(icon: Icon(Icons.person_outlined), selectedIcon: Icon(Icons.person, color: kPrimary), label: 'Profile'),
        ],
      ),
    );
  }
}

// ============================================================
// HOME SCREEN
// ============================================================
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final user = _auth.currentUser;
    return Scaffold(
      backgroundColor: kBg,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            floating: true,
            backgroundColor: kPrimary,
            expandedHeight: 140,
            flexibleSpace: FlexibleSpaceBar(
              background: Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(colors: [kPrimary, kPrimaryL], begin: Alignment.topLeft, end: Alignment.bottomRight),
                ),
                padding: const EdgeInsets.fromLTRB(20, 60, 20, 20),
                child: Row(children: [
                  buildAvatar(user?.photoURL, r: 26),
                  const SizedBox(width: 12),
                  Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisAlignment: MainAxisAlignment.center, children: [
                    const Text('Welcome back,', style: TextStyle(color: Colors.white70, fontSize: 13)),
                    Text(user?.displayName ?? 'Student', style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w700)),
                  ])),
                  IconButton(
                    icon: const Icon(Icons.search, color: Colors.white),
                    onPressed: () => Navigator.push(context, _route(const SearchScreen())),
                  ),
                ]),
              ),
            ),
          ),
          SliverToBoxAdapter(child: Column(children: [
            const SizedBox(height: 12),
            _buildQuickActions(context),
            const SizedBox(height: 16),
            buildSectionHeader('Recent Team Posts'),
          ])),
          SliverFillRemaining(
            child: StreamBuilder<QuerySnapshot>(
              stream: _firestore.collection('team_posts').orderBy('createdAt', descending: true).limit(5).snapshots(),
              builder: (ctx, snap) {
                if (snap.connectionState == ConnectionState.waiting) {
                  return ListView.builder(itemCount: 3, itemBuilder: (_, __) => buildShimmerCard());
                }
                final docs = snap.data?.docs ?? [];
                if (docs.isEmpty) return buildEmpty('No posts yet. Be the first!');
                return ListView.builder(
                  itemCount: docs.length,
                  itemBuilder: (_, i) => _buildPostCard(context, docs[i]),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQuickActions(BuildContext ctx) => Padding(
    padding: const EdgeInsets.symmetric(horizontal: 16),
    child: Row(children: [
      _quickAction(ctx, Icons.add_circle, 'Post Team', const CreateTeamPostScreen()),
      const SizedBox(width: 12),
      _quickAction(ctx, Icons.category, 'Categories', const CategoriesScreen()),
      const SizedBox(width: 12),
      _quickAction(ctx, Icons.workspace_premium, 'My Teams', const AcceptedTeamsScreen()),
      const SizedBox(width: 12),
      _quickAction(ctx, Icons.work_outline, 'Portfolio', const PortfolioScreen()),
    ]),
  );

  Widget _quickAction(BuildContext ctx, IconData icon, String label, Widget dest) => Expanded(
    child: GestureDetector(
      onTap: () => Navigator.push(ctx, _route(dest)),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(14),
          boxShadow: [BoxShadow(color: kPrimary.withOpacity(0.07), blurRadius: 8, offset: const Offset(0, 2))],
        ),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, color: kPrimary, size: 26),
          const SizedBox(height: 6),
          Text(label, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: kText), textAlign: TextAlign.center),
        ]),
      ),
    ),
  );

  Widget _buildPostCard(BuildContext ctx, DocumentSnapshot doc) {
    final d = doc.data() as Map<String, dynamic>;
    return buildCardTile(
      onTap: () => Navigator.push(ctx, _route(PostDetailsScreen(postId: doc.id, data: d))),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          buildAvatar(d['authorPhoto']),
          const SizedBox(width: 10),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(d['authorName'] ?? '', style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13)),
            Text(d['createdAt'] != null ? timeago.format((d['createdAt'] as Timestamp).toDate()) : '',
                style: const TextStyle(fontSize: 11, color: kTextSub)),
          ])),
          buildTag(d['category'] ?? 'General'),
        ]),
        const SizedBox(height: 10),
        Text(d['title'] ?? '', style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
        const SizedBox(height: 4),
        Text(d['description'] ?? '', maxLines: 2, overflow: TextOverflow.ellipsis, style: const TextStyle(color: kTextSub, fontSize: 13)),
      ]),
    );
  }
}

// ============================================================
// TEAM POSTS FEED SCREEN
// ============================================================
class TeamPostsFeedScreen extends StatefulWidget {
  const TeamPostsFeedScreen({super.key});
  @override
  State<TeamPostsFeedScreen> createState() => _TeamPostsFeedScreenState();
}

class _TeamPostsFeedScreenState extends State<TeamPostsFeedScreen> {
  String _filter = 'All';
  final _categories = ['All', 'Hackathon', 'Project', 'Research', 'Competition', 'Startup', 'Study'];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Team Posts'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () => Navigator.push(context, _route(const CreateTeamPostScreen())),
          ),
        ],
      ),
      body: Column(children: [
        SizedBox(
          height: 48,
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            itemCount: _categories.length,
            itemBuilder: (_, i) {
              final selected = _filter == _categories[i];
              return GestureDetector(
                onTap: () => setState(() => _filter = _categories[i]),
                child: Container(
                  margin: const EdgeInsets.only(right: 8),
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  decoration: BoxDecoration(
                    color: selected ? kPrimary : Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: selected ? kPrimary : kDivider),
                  ),
                  child: Center(child: Text(_categories[i],
                      style: TextStyle(color: selected ? Colors.white : kText, fontWeight: FontWeight.w600, fontSize: 13))),
                ),
              );
            },
          ),
        ),
        Expanded(
          child: StreamBuilder<QuerySnapshot>(
            stream: _filter == 'All'
                ? postsStream()
                : _firestore.collection('team_posts').where('category', isEqualTo: _filter).orderBy('createdAt', descending: true).snapshots(),
            builder: (ctx, snap) {
              if (snap.connectionState == ConnectionState.waiting) {
                return ListView.builder(itemCount: 5, itemBuilder: (_, __) => buildShimmerCard());
              }
              if (snap.hasError) return buildError('Something went wrong');
              final docs = snap.data?.docs ?? [];
              if (docs.isEmpty) return buildEmpty('No posts in this category');
              return ListView.builder(
                padding: const EdgeInsets.only(top: 4, bottom: 80),
                itemCount: docs.length,
                itemBuilder: (_, i) => _PostCard(doc: docs[i]),
              );
            },
          ),
        ),
      ]),
    );
  }
}

class _PostCard extends StatelessWidget {
  final DocumentSnapshot doc;
  const _PostCard({required this.doc});

  @override
  Widget build(BuildContext context) {
    final d = doc.data() as Map<String, dynamic>;
    final isOwner = _auth.currentUser?.uid == d['authorId'];
    return buildCardTile(
      onTap: () => Navigator.push(context, _route(PostDetailsScreen(postId: doc.id, data: d))),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          buildAvatar(d['authorPhoto']),
          const SizedBox(width: 10),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(d['authorName'] ?? '', style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13)),
            if (d['createdAt'] != null)
              Text(timeago.format((d['createdAt'] as Timestamp).toDate()),
                  style: const TextStyle(fontSize: 11, color: kTextSub)),
          ])),
          buildTag(d['category'] ?? 'General'),
          if (isOwner) PopupMenuButton<String>(
            onSelected: (v) {
              if (v == 'edit') Navigator.push(context, _route(EditTeamPostScreen(postId: doc.id, data: d)));
              if (v == 'delete') _confirmDelete(context);
            },
            itemBuilder: (_) => const [
              PopupMenuItem(value: 'edit', child: Text('Edit')),
              PopupMenuItem(value: 'delete', child: Text('Delete', style: TextStyle(color: kError))),
            ],
          ),
        ]),
        const SizedBox(height: 10),
        Text(d['title'] ?? '', style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
        const SizedBox(height: 4),
        Text(d['description'] ?? '', maxLines: 2, overflow: TextOverflow.ellipsis,
            style: const TextStyle(color: kTextSub, fontSize: 13)),
        const SizedBox(height: 10),
        Row(children: [
          const Icon(Icons.people_outline, size: 14, color: kTextSub),
          const SizedBox(width: 4),
          Text('${d['teamSize'] ?? 1} members needed', style: const TextStyle(fontSize: 12, color: kTextSub)),
          const Spacer(),
          const Icon(Icons.send_outlined, size: 14, color: kTextSub),
          const SizedBox(width: 4),
          Text('${d['requestCount'] ?? 0} requests', style: const TextStyle(fontSize: 12, color: kTextSub)),
        ]),
      ]),
    );
  }

  void _confirmDelete(BuildContext ctx) => showDialog(
    context: ctx,
    builder: (_) => AlertDialog(
      title: const Text('Delete Post'),
      content: const Text('Are you sure you want to delete this post?'),
      actions: [
        TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
        ElevatedButton(
          onPressed: () async { Navigator.pop(ctx); await deletePost(doc.id); },
          style: ElevatedButton.styleFrom(backgroundColor: kError),
          child: const Text('Delete'),
        ),
      ],
    ),
  );
}

// ============================================================
// POST DETAILS SCREEN
// ============================================================
class PostDetailsScreen extends StatelessWidget {
  final String postId;
  final Map<String, dynamic> data;
  const PostDetailsScreen({super.key, required this.postId, required this.data});

  @override
  Widget build(BuildContext context) {
    final isOwner = _auth.currentUser?.uid == data['authorId'];
    return Scaffold(
      appBar: AppBar(
        title: const Text('Post Details'),
        actions: [
          if (isOwner)
            IconButton(
              icon: const Icon(Icons.edit),
              onPressed: () => Navigator.push(context, _route(EditTeamPostScreen(postId: postId, data: data))),
            ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Row(children: [
                  buildAvatar(data['authorPhoto'], r: 28),
                  const SizedBox(width: 12),
                  Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(data['authorName'] ?? '', style: const TextStyle(fontWeight: FontWeight.w700)),
                    if (data['createdAt'] != null)
                      Text(DateFormat('MMM d, yyyy').format((data['createdAt'] as Timestamp).toDate()),
                          style: const TextStyle(color: kTextSub, fontSize: 12)),
                  ]),
                  const Spacer(),
                  buildTag(data['category'] ?? 'General'),
                ]),
                const SizedBox(height: 14),
                Text(data['title'] ?? '', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w800)),
                const SizedBox(height: 10),
                Text(data['description'] ?? '', style: const TextStyle(fontSize: 14, color: kTextSub, height: 1.5)),
                const SizedBox(height: 14),
                kDividerWidget(),
                const SizedBox(height: 12),
                _infoRow(Icons.people_outline, 'Team size: ${data['teamSize'] ?? 1}'),
                const SizedBox(height: 8),
                _infoRow(Icons.code, 'Skills: ${(data['skills'] as List?)?.join(', ') ?? 'Any'}'),
                const SizedBox(height: 8),
                _infoRow(Icons.calendar_today, 'Deadline: ${data['deadline'] ?? 'Open'}'),
              ]),
            ),
          ),
          const SizedBox(height: 16),
          if (!isOwner)
            buildGradientButton(
              label: 'Request to Join',
              icon: Icons.send,
              onTap: () async {
                await sendJoinRequest(postId, data['title'] ?? '', data['authorId'] ?? '');
                if (context.mounted) showSnack(context, 'Request sent!');
              },
            ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            onPressed: () {
              final chatId = [_auth.currentUser!.uid, data['authorId']].toList()..sort();
              Navigator.push(context, _route(ChatScreen(
                chatId: chatId.join('_'),
                otherUserId: data['authorId'],
                otherUserName: data['authorName'] ?? 'User',
                otherUserPhoto: data['authorPhoto'] ?? '',
              )));
            },
            icon: const Icon(Icons.chat_outlined),
            label: const Text('Message Author'),
            style: OutlinedButton.styleFrom(
              minimumSize: const Size(double.infinity, 50),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ]),
      ),
    );
  }

  Widget _infoRow(IconData icon, String text) => Row(children: [
    Icon(icon, size: 16, color: kPrimary),
    const SizedBox(width: 8),
    Expanded(child: Text(text, style: const TextStyle(fontSize: 13, color: kText))),
  ]);
}

// ============================================================
// CREATE TEAM POST SCREEN
// ============================================================
class CreateTeamPostScreen extends StatefulWidget {
  const CreateTeamPostScreen({super.key});
  @override
  State<CreateTeamPostScreen> createState() => _CreateTeamPostScreenState();
}

class _CreateTeamPostScreenState extends State<CreateTeamPostScreen> {
  final _formKey    = GlobalKey<FormState>();
  final _titleCtrl  = TextEditingController();
  final _descCtrl   = TextEditingController();
  final _skillCtrl  = TextEditingController();
  final _deadCtrl   = TextEditingController();
  String _category  = 'Hackathon';
  int    _teamSize  = 2;
  bool   _loading   = false;
  final  _skills    = <String>[];
  final _cats       = ['Hackathon', 'Project', 'Research', 'Competition', 'Startup', 'Study'];

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _loading = true);
    try {
      await createPost({
        'title': _titleCtrl.text.trim(),
        'description': _descCtrl.text.trim(),
        'category': _category,
        'teamSize': _teamSize,
        'skills': _skills,
        'deadline': _deadCtrl.text.trim(),
      });
      if (mounted) { Navigator.pop(context); showSnack(context, 'Post created!'); }
    } catch (e) {
      if (mounted) showSnack(context, 'Error: $e', error: true);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _addSkill() {
    final s = _skillCtrl.text.trim();
    if (s.isNotEmpty && !_skills.contains(s)) {
      setState(() { _skills.add(s); _skillCtrl.clear(); });
    }
  }

  @override
  void dispose() { _titleCtrl.dispose(); _descCtrl.dispose(); _skillCtrl.dispose(); _deadCtrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Create Team Post')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            _label('Title'),
            TextFormField(controller: _titleCtrl, decoration: const InputDecoration(hintText: 'e.g. Need Backend Dev for Hackathon'),
                validator: (v) => (v == null || v.isEmpty) ? 'Required' : null),
            const SizedBox(height: 14),
            _label('Description'),
            TextFormField(controller: _descCtrl, maxLines: 4, decoration: const InputDecoration(hintText: 'Describe the project...'),
                validator: (v) => (v == null || v.isEmpty) ? 'Required' : null),
            const SizedBox(height: 14),
            _label('Category'),
            DropdownButtonFormField<String>(
              value: _category,
              decoration: const InputDecoration(),
              items: _cats.map((c) => DropdownMenuItem(value: c, child: Text(c))).toList(),
              onChanged: (v) => setState(() => _category = v!),
            ),
            const SizedBox(height: 14),
            _label('Team Size Needed'),
            Row(children: [
              IconButton(icon: const Icon(Icons.remove_circle_outline), onPressed: () => setState(() { if (_teamSize > 1) _teamSize--; })),
              Text('$_teamSize', style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
              IconButton(icon: const Icon(Icons.add_circle_outline), onPressed: () => setState(() => _teamSize++)),
            ]),
            const SizedBox(height: 14),
            _label('Required Skills'),
            Row(children: [
              Expanded(child: TextFormField(controller: _skillCtrl, decoration: const InputDecoration(hintText: 'Add a skill...'))),
              const SizedBox(width: 8),
              IconButton(onPressed: _addSkill, icon: const Icon(Icons.add_circle, color: kPrimary)),
            ]),
            const SizedBox(height: 8),
            Wrap(spacing: 8, children: _skills.map((s) => Chip(
              label: Text(s),
              deleteIcon: const Icon(Icons.close, size: 14),
              onDeleted: () => setState(() => _skills.remove(s)),
              backgroundColor: kPrimary.withOpacity(0.1),
              labelStyle: const TextStyle(color: kPrimary),
            )).toList()),
            const SizedBox(height: 14),
            _label('Deadline (optional)'),
            TextFormField(controller: _deadCtrl, decoration: const InputDecoration(hintText: 'e.g. Dec 15, 2025')),
            const SizedBox(height: 24),
            _loading ? buildLoading() : buildGradientButton(label: 'Post Team', onTap: _submit, icon: Icons.send),
          ]),
        ),
      ),
    );
  }

  Widget _label(String t) => Padding(
    padding: const EdgeInsets.only(bottom: 6),
    child: Text(t, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13, color: kText)),
  );
}

// ============================================================
// EDIT TEAM POST SCREEN
// ============================================================
class EditTeamPostScreen extends StatefulWidget {
  final String postId;
  final Map<String, dynamic> data;
  const EditTeamPostScreen({super.key, required this.postId, required this.data});
  @override
  State<EditTeamPostScreen> createState() => _EditTeamPostScreenState();
}

class _EditTeamPostScreenState extends State<EditTeamPostScreen> {
  late final TextEditingController _titleCtrl;
  late final TextEditingController _descCtrl;
  late final TextEditingController _deadCtrl;
  late String _category;
  late int    _teamSize;
  bool _loading = false;
  final _cats = ['Hackathon', 'Project', 'Research', 'Competition', 'Startup', 'Study'];

  @override
  void initState() {
    super.initState();
    _titleCtrl = TextEditingController(text: widget.data['title'] ?? '');
    _descCtrl  = TextEditingController(text: widget.data['description'] ?? '');
    _deadCtrl  = TextEditingController(text: widget.data['deadline'] ?? '');
    _category  = widget.data['category'] ?? 'Hackathon';
    _teamSize  = widget.data['teamSize'] ?? 2;
  }

  Future<void> _update() async {
    setState(() => _loading = true);
    try {
      await updatePost(widget.postId, {
        'title': _titleCtrl.text.trim(),
        'description': _descCtrl.text.trim(),
        'category': _category,
        'teamSize': _teamSize,
        'deadline': _deadCtrl.text.trim(),
      });
      if (mounted) { Navigator.pop(context); showSnack(context, 'Post updated!'); }
    } catch (e) {
      if (mounted) showSnack(context, 'Error: $e', error: true);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  void dispose() { _titleCtrl.dispose(); _descCtrl.dispose(); _deadCtrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Edit Post')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Title', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          TextFormField(controller: _titleCtrl, decoration: const InputDecoration()),
          const SizedBox(height: 14),
          const Text('Description', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          TextFormField(controller: _descCtrl, maxLines: 4, decoration: const InputDecoration()),
          const SizedBox(height: 14),
          const Text('Category', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          DropdownButtonFormField<String>(
            value: _cats.contains(_category) ? _category : _cats.first,
            decoration: const InputDecoration(),
            items: _cats.map((c) => DropdownMenuItem(value: c, child: Text(c))).toList(),
            onChanged: (v) => setState(() => _category = v!),
          ),
          const SizedBox(height: 14),
          const Text('Team Size', style: TextStyle(fontWeight: FontWeight.w600)),
          Row(children: [
            IconButton(icon: const Icon(Icons.remove_circle_outline), onPressed: () => setState(() { if (_teamSize > 1) _teamSize--; })),
            Text('$_teamSize', style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
            IconButton(icon: const Icon(Icons.add_circle_outline), onPressed: () => setState(() => _teamSize++)),
          ]),
          const SizedBox(height: 14),
          const Text('Deadline', style: TextStyle(fontWeight: FontWeight.w600)),
          const SizedBox(height: 6),
          TextFormField(controller: _deadCtrl, decoration: const InputDecoration()),
          const SizedBox(height: 24),
          _loading ? buildLoading() : buildGradientButton(label: 'Update Post', onTap: _update, icon: Icons.save),
        ]),
      ),
    );
  }
}

// ============================================================
// CATEGORIES SCREEN
// ============================================================
class CategoriesScreen extends StatelessWidget {
  const CategoriesScreen({super.key});

  static const _cats = [
    {'label': 'Hackathon',   'icon': Icons.code,              'color': Color(0xFF1565C0)},
    {'label': 'Project',     'icon': Icons.folder_open,       'color': Color(0xFF00897B)},
    {'label': 'Research',    'icon': Icons.science,           'color': Color(0xFF6A1B9A)},
    {'label': 'Competition', 'icon': Icons.emoji_events,      'color': Color(0xFFE65100)},
    {'label': 'Startup',     'icon': Icons.rocket_launch,     'color': Color(0xFFC62828)},
    {'label': 'Study',       'icon': Icons.menu_book,         'color': Color(0xFF2E7D32)},
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Categories')),
      body: GridView.builder(
        padding: const EdgeInsets.all(16),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 2, mainAxisSpacing: 12, crossAxisSpacing: 12, childAspectRatio: 1.3),
        itemCount: _cats.length,
        itemBuilder: (_, i) {
          final cat = _cats[i];
          return GestureDetector(
            onTap: () => Navigator.push(context, _route(_CategoryPostsScreen(category: cat['label'] as String))),
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(colors: [(cat['color'] as Color), (cat['color'] as Color).withOpacity(0.7)]),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                Icon(cat['icon'] as IconData, size: 40, color: Colors.white),
                const SizedBox(height: 10),
                Text(cat['label'] as String, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700, fontSize: 15)),
              ]),
            ),
          );
        },
      ),
    );
  }
}

class _CategoryPostsScreen extends StatelessWidget {
  final String category;
  const _CategoryPostsScreen({required this.category});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(category)),
      body: StreamBuilder<QuerySnapshot>(
        stream: _firestore.collection('team_posts').where('category', isEqualTo: category).orderBy('createdAt', descending: true).snapshots(),
        builder: (ctx, snap) {
          if (snap.connectionState == ConnectionState.waiting) return buildLoading();
          final docs = snap.data?.docs ?? [];
          if (docs.isEmpty) return buildEmpty('No $category posts yet');
          return ListView.builder(
            itemCount: docs.length,
            itemBuilder: (_, i) => _PostCard(doc: docs[i]),
          );
        },
      ),
    );
  }
}

// ============================================================
// SEARCH SCREEN
// ============================================================
class SearchScreen extends StatefulWidget {
  const SearchScreen({super.key});
  @override
  State<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends State<SearchScreen> {
  final _ctrl = TextEditingController();
  List<DocumentSnapshot> _results = [];
  bool _searching = false;

  Future<void> _search(String q) async {
    if (q.trim().isEmpty) { setState(() => _results = []); return; }
    setState(() => _searching = true);
    final snap = await _firestore.collection('team_posts').orderBy('title').startAt([q]).endAt(['$q']).limit(20).get();
    if (mounted) setState(() { _results = snap.docs; _searching = false; });
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: TextField(
          controller: _ctrl,
          autofocus: true,
          style: const TextStyle(color: Colors.white),
          decoration: const InputDecoration(
            hintText: 'Search posts...',
            hintStyle: TextStyle(color: Colors.white54),
            border: InputBorder.none,
            filled: false,
          ),
          onChanged: _search,
        ),
      ),
      body: _searching
          ? buildLoading()
          : _results.isEmpty
              ? buildEmpty('Search for team posts', icon: Icons.search)
              : ListView.builder(
                  itemCount: _results.length,
                  itemBuilder: (_, i) => _PostCard(doc: _results[i]),
                ),
    );
  }
}

// ============================================================
// CHAT LIST SCREEN
// ============================================================
class ChatListScreen extends StatelessWidget {
  const ChatListScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final uid = _auth.currentUser?.uid ?? '';
    return Scaffold(
      appBar: AppBar(title: const Text('Messages')),
      body: StreamBuilder<QuerySnapshot>(
        stream: _firestore.collection('messages').where('senderId', isEqualTo: uid).snapshots(),
        builder: (ctx, snap) {
          if (snap.connectionState == ConnectionState.waiting) return buildLoading();
          final msgs = snap.data?.docs ?? [];

          // Build unique chat list
          final Map<String, Map<String, dynamic>> chats = {};
          for (final doc in msgs) {
            final d = doc.data() as Map<String, dynamic>;
            final chatId = d['chatId'] as String? ?? '';
            if (!chats.containsKey(chatId)) chats[chatId] = d;
          }

          // Also listen for received messages
          return StreamBuilder<QuerySnapshot>(
            stream: _firestore.collection('messages').where('receiverId', isEqualTo: uid).snapshots(),
            builder: (ctx2, snap2) {
              final msgs2 = snap2.data?.docs ?? [];
              for (final doc in msgs2) {
                final d = doc.data() as Map<String, dynamic>;
                final chatId = d['chatId'] as String? ?? '';
                if (!chats.containsKey(chatId)) chats[chatId] = d;
              }

              final chatList = chats.values.toList();
              if (chatList.isEmpty) return buildEmpty('No conversations yet', icon: Icons.chat_bubble_outline);

              return ListView.separated(
                itemCount: chatList.length,
                separatorBuilder: (_, __) => kDividerWidget(),
                itemBuilder: (_, i) {
                  final c = chatList[i];
                  final isMe = c['senderId'] == uid;
                  final otherId = isMe ? c['receiverId'] as String : c['senderId'] as String;
                  final otherName = isMe ? 'User' : (c['senderName'] as String? ?? 'User');
                  final otherPhoto = isMe ? '' : (c['senderPhoto'] as String? ?? '');
                  final chatId = c['chatId'] as String? ?? '';
                  return ListTile(
                    leading: buildAvatar(otherPhoto),
                    title: Text(otherName, style: const TextStyle(fontWeight: FontWeight.w600)),
                    subtitle: Text(c['text'] ?? '', maxLines: 1, overflow: TextOverflow.ellipsis),
                    trailing: c['sentAt'] != null
                        ? Text(timeago.format((c['sentAt'] as Timestamp).toDate()), style: const TextStyle(fontSize: 11, color: kTextSub))
                        : null,
                    onTap: () => Navigator.push(context, _route(ChatScreen(
                      chatId: chatId, otherUserId: otherId,
                      otherUserName: otherName, otherUserPhoto: otherPhoto,
                    ))),
                  );
                },
              );
            },
          );
        },
      ),
    );
  }
}

// ============================================================
// CHAT SCREEN
// ============================================================
class ChatScreen extends StatefulWidget {
  final String chatId, otherUserId, otherUserName, otherUserPhoto;
  const ChatScreen({super.key, required this.chatId, required this.otherUserId, required this.otherUserName, required this.otherUserPhoto});
  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _ctrl       = TextEditingController();
  final _scrollCtrl = ScrollController();

  void _send() async {
    final text = _ctrl.text.trim();
    if (text.isEmpty) return;
    _ctrl.clear();
    await sendMessage(widget.chatId, text, widget.otherUserId);
    if (_scrollCtrl.hasClients) {
      _scrollCtrl.animateTo(0, duration: const Duration(milliseconds: 300), curve: Curves.easeOut);
    }
  }

  @override
  void dispose() { _ctrl.dispose(); _scrollCtrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    final uid = _auth.currentUser?.uid ?? '';
    return Scaffold(
      appBar: AppBar(
        title: Row(children: [
          buildAvatar(widget.otherUserPhoto, r: 18),
          const SizedBox(width: 10),
          Text(widget.otherUserName),
        ]),
      ),
      body: Column(children: [
        Expanded(
          child: StreamBuilder<QuerySnapshot>(
            stream: chatStream(widget.chatId),
            builder: (ctx, snap) {
              if (snap.connectionState == ConnectionState.waiting) return buildLoading();
              final docs = snap.data?.docs ?? [];
              if (docs.isEmpty) return buildEmpty('Start chatting!', icon: Icons.chat_outlined);
              return ListView.builder(
                reverse: true,
                controller: _scrollCtrl,
                padding: const EdgeInsets.all(12),
                itemCount: docs.length,
                itemBuilder: (_, i) {
                  final d = docs[i].data() as Map<String, dynamic>;
                  final isMe = d['senderId'] == uid;
                  return Align(
                    alignment: isMe ? Alignment.centerRight : Alignment.centerLeft,
                    child: Container(
                      margin: const EdgeInsets.only(bottom: 8),
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                      constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.72),
                      decoration: BoxDecoration(
                        color: isMe ? kPrimary : Colors.white,
                        borderRadius: BorderRadius.only(
                          topLeft: const Radius.circular(16),
                          topRight: const Radius.circular(16),
                          bottomLeft: Radius.circular(isMe ? 16 : 4),
                          bottomRight: Radius.circular(isMe ? 4 : 16),
                        ),
                        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.06), blurRadius: 4, offset: const Offset(0, 2))],
                      ),
                      child: Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
                        Text(d['text'] ?? '', style: TextStyle(color: isMe ? Colors.white : kText, fontSize: 14)),
                        if (d['sentAt'] != null)
                          Text(
                            DateFormat('HH:mm').format((d['sentAt'] as Timestamp).toDate()),
                            style: TextStyle(fontSize: 10, color: isMe ? Colors.white54 : kTextSub),
                          ),
                      ]),
                    ),
                  );
                },
              );
            },
          ),
        ),
        Container(
          color: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: SafeArea(
            top: false,
            child: Row(children: [
              Expanded(
                child: TextField(
                  controller: _ctrl,
                  decoration: InputDecoration(
                    hintText: 'Type a message...',
                    filled: true,
                    fillColor: kBg,
                    contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(24), borderSide: BorderSide.none),
                    enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(24), borderSide: BorderSide.none),
                  ),
                  textInputAction: TextInputAction.send,
                  onSubmitted: (_) => _send(),
                ),
              ),
              const SizedBox(width: 8),
              GestureDetector(
                onTap: _send,
                child: Container(
                  padding: const EdgeInsets.all(12),
                  decoration: const BoxDecoration(color: kPrimary, shape: BoxShape.circle),
                  child: const Icon(Icons.send, color: Colors.white, size: 20),
                ),
              ),
            ]),
          ),
        ),
      ]),
    );
  }
}

// ============================================================
// NOTIFICATIONS SCREEN
// ============================================================
class NotificationsScreen extends StatelessWidget {
  const NotificationsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Notifications')),
      body: StreamBuilder<QuerySnapshot>(
        stream: myRequestsStream(),
        builder: (ctx, snap) {
          if (snap.connectionState == ConnectionState.waiting) return buildLoading();
          final docs = snap.data?.docs ?? [];
          if (docs.isEmpty) return buildEmpty('No notifications yet', icon: Icons.notifications_none);
          return ListView.separated(
            itemCount: docs.length,
            separatorBuilder: (_, __) => kDividerWidget(),
            itemBuilder: (_, i) {
              final d = docs[i].data() as Map<String, dynamic>;
              final status = d['status'] as String? ?? 'pending';
              return ListTile(
                leading: buildAvatar(d['senderPhoto']),
                title: Text('${d['senderName']} wants to join "${d['postTitle']}"', style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13)),
                subtitle: Text(d['createdAt'] != null ? timeago.format((d['createdAt'] as Timestamp).toDate()) : ''),
                trailing: status == 'pending'
                    ? Row(mainAxisSize: MainAxisSize.min, children: [
                        IconButton(
                          icon: const Icon(Icons.check_circle, color: kSuccess),
                          onPressed: () => updateRequest(docs[i].id, 'accepted'),
                        ),
                        IconButton(
                          icon: const Icon(Icons.cancel, color: kError),
                          onPressed: () => updateRequest(docs[i].id, 'rejected'),
                        ),
                      ])
                    : buildTag(status, bg: status == 'accepted' ? kSuccess : kError),
              );
            },
          );
        },
      ),
    );
  }
}

// ============================================================
// JOIN REQUESTS SCREEN
// ============================================================
class JoinRequestsScreen extends StatelessWidget {
  const JoinRequestsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Join Requests')),
      body: StreamBuilder<QuerySnapshot>(
        stream: myRequestsStream(),
        builder: (ctx, snap) {
          if (snap.connectionState == ConnectionState.waiting) return buildLoading();
          final docs = snap.data?.docs ?? [];
          if (docs.isEmpty) return buildEmpty('No join requests yet');
          return ListView.builder(
            itemCount: docs.length,
            itemBuilder: (_, i) {
              final d = docs[i].data() as Map<String, dynamic>;
              final status = d['status'] as String? ?? 'pending';
              return buildCardTile(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Row(children: [
                    buildAvatar(d['senderPhoto']),
                    const SizedBox(width: 10),
                    Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text(d['senderName'] ?? '', style: const TextStyle(fontWeight: FontWeight.w700)),
                      Text('For: ${d['postTitle'] ?? ''}', style: const TextStyle(color: kTextSub, fontSize: 12)),
                    ])),
                    buildTag(status, bg: status == 'accepted' ? kSuccess : status == 'rejected' ? kError : kWarning),
                  ]),
                  if (status == 'pending') ...[
                    const SizedBox(height: 12),
                    Row(children: [
                      Expanded(child: OutlinedButton(
                        onPressed: () => updateRequest(docs[i].id, 'rejected'),
                        style: OutlinedButton.styleFrom(foregroundColor: kError, side: const BorderSide(color: kError)),
                        child: const Text('Decline'),
                      )),
                      const SizedBox(width: 10),
                      Expanded(child: ElevatedButton(
                        onPressed: () => updateRequest(docs[i].id, 'accepted'),
                        child: const Text('Accept'),
                      )),
                    ]),
                  ],
                ]),
              );
            },
          );
        },
      ),
    );
  }
}

// ============================================================
// ACCEPTED TEAMS SCREEN
// ============================================================
class AcceptedTeamsScreen extends StatelessWidget {
  const AcceptedTeamsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('My Teams')),
      body: StreamBuilder<QuerySnapshot>(
        stream: acceptedTeamsStream(),
        builder: (ctx, snap) {
          if (snap.connectionState == ConnectionState.waiting) return buildLoading();
          final docs = snap.data?.docs ?? [];
          if (docs.isEmpty) return buildEmpty('You haven\'t joined any team yet', icon: Icons.groups_outlined);
          return ListView.builder(
            padding: const EdgeInsets.all(12),
            itemCount: docs.length,
            itemBuilder: (_, i) {
              final d = docs[i].data() as Map<String, dynamic>;
              return buildCardTile(
                child: Row(children: [
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(color: kPrimary.withOpacity(0.1), borderRadius: BorderRadius.circular(12)),
                    child: const Icon(Icons.groups, color: kPrimary),
                  ),
                  const SizedBox(width: 12),
                  Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(d['postTitle'] ?? '', style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 14)),
                    Text('Accepted', style: const TextStyle(color: kSuccess, fontWeight: FontWeight.w600, fontSize: 12)),
                  ])),
                  buildTag('Active', bg: kSuccess),
                ]),
              );
            },
          );
        },
      ),
    );
  }
}

// ============================================================
// PROFILE SCREEN
// ============================================================
class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final user = _auth.currentUser;
    if (user == null) return const LoginScreen();
    return Scaffold(
      body: StreamBuilder<DocumentSnapshot>(
        stream: _firestore.collection('users').doc(user.uid).snapshots(),
        builder: (ctx, snap) {
          final data = snap.data?.data() as Map<String, dynamic>? ?? {};
          return CustomScrollView(
            slivers: [
              SliverAppBar(
                expandedHeight: 200,
                pinned: true,
                backgroundColor: kPrimary,
                actions: [
                  IconButton(icon: const Icon(Icons.edit), onPressed: () => Navigator.push(context, _route(EditProfileScreen(data: data)))),
                  IconButton(icon: const Icon(Icons.settings), onPressed: () => Navigator.push(context, _route(const SettingsScreen()))),
                ],
                flexibleSpace: FlexibleSpaceBar(
                  background: Container(
                    decoration: const BoxDecoration(gradient: LinearGradient(colors: [kPrimary, kPrimaryL])),
                    child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                      const SizedBox(height: 40),
                      buildAvatar(data['photoUrl'] as String?, r: 44),
                      const SizedBox(height: 12),
                      Text(data['name'] ?? user.displayName ?? 'User',
                          style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w800)),
                      Text(data['major'] ?? data['email'] ?? '', style: const TextStyle(color: Colors.white70, fontSize: 13)),
                    ]),
                  ),
                ),
              ),
              SliverToBoxAdapter(child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  if ((data['bio'] as String? ?? '').isNotEmpty) ...[
                    const Text('About', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
                    const SizedBox(height: 6),
                    Text(data['bio'] as String? ?? '', style: const TextStyle(color: kTextSub, height: 1.5)),
                    const SizedBox(height: 14),
                  ],
                  if ((data['skills'] as List?)?.isNotEmpty == true) ...[
                    const Text('Skills', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
                    const SizedBox(height: 8),
                    Wrap(spacing: 8, runSpacing: 8,
                      children: (data['skills'] as List).map((s) => buildTag(s.toString())).toList()),
                    const SizedBox(height: 14),
                  ],
                  Row(children: [
                    Expanded(child: _profileAction(context, Icons.work_outline, 'Portfolio', const PortfolioScreen())),
                    const SizedBox(width: 12),
                    Expanded(child: _profileAction(context, Icons.groups, 'My Teams', const AcceptedTeamsScreen())),
                  ]),
                  const SizedBox(height: 12),
                  Row(children: [
                    Expanded(child: _profileAction(context, Icons.send, 'Requests', const JoinRequestsScreen())),
                    const SizedBox(width: 12),
                    Expanded(child: _profileAction(context, Icons.help_outline, 'Help', const HelpSupportScreen())),
                  ]),
                  const SizedBox(height: 20),
                  buildGradientButton(
                    label: 'Sign Out',
                    icon: Icons.logout,
                    onTap: () async {
                      await signOut();
                      if (context.mounted) Navigator.pushAndRemoveUntil(context, _route(const LoginScreen()), (_) => false);
                    },
                  ),
                ]),
              )),
            ],
          );
        },
      ),
    );
  }

  Widget _profileAction(BuildContext ctx, IconData icon, String label, Widget dest) => GestureDetector(
    onTap: () => Navigator.push(ctx, _route(dest)),
    child: Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        boxShadow: [BoxShadow(color: kPrimary.withOpacity(0.07), blurRadius: 8)],
      ),
      child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
        Icon(icon, color: kPrimary, size: 20),
        const SizedBox(width: 8),
        Text(label, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13)),
      ]),
    ),
  );
}

// ============================================================
// EDIT PROFILE SCREEN
// ============================================================
class EditProfileScreen extends StatefulWidget {
  final Map<String, dynamic> data;
  const EditProfileScreen({super.key, required this.data});
  @override
  State<EditProfileScreen> createState() => _EditProfileScreenState();
}

class _EditProfileScreenState extends State<EditProfileScreen> {
  late final TextEditingController _nameCtrl;
  late final TextEditingController _bioCtrl;
  late final TextEditingController _majorCtrl;
  late final TextEditingController _yearCtrl;
  late final TextEditingController _skillCtrl;
  late List<String> _skills;
  bool _loading = false;
  File? _imageFile;

  @override
  void initState() {
    super.initState();
    _nameCtrl  = TextEditingController(text: widget.data['name'] ?? '');
    _bioCtrl   = TextEditingController(text: widget.data['bio'] ?? '');
    _majorCtrl = TextEditingController(text: widget.data['major'] ?? '');
    _yearCtrl  = TextEditingController(text: widget.data['year'] ?? '');
    _skillCtrl = TextEditingController();
    _skills    = List<String>.from(widget.data['skills'] ?? []);
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final xfile = await picker.pickImage(source: ImageSource.gallery, imageQuality: 70);
    if (xfile != null) setState(() => _imageFile = File(xfile.path));
  }

  Future<void> _save() async {
    setState(() => _loading = true);
    try {
      String? photoUrl = widget.data['photoUrl'];
      if (_imageFile != null) {
        photoUrl = await uploadImage(_imageFile!, 'avatars/${_auth.currentUser!.uid}.jpg');
      }
      await updateUserDoc(_auth.currentUser!.uid, {
        'name': _nameCtrl.text.trim(),
        'bio': _bioCtrl.text.trim(),
        'major': _majorCtrl.text.trim(),
        'year': _yearCtrl.text.trim(),
        'skills': _skills,
        if (photoUrl != null) 'photoUrl': photoUrl,
      });
      await _auth.currentUser!.updateDisplayName(_nameCtrl.text.trim());
      if (mounted) { Navigator.pop(context); showSnack(context, 'Profile updated!'); }
    } catch (e) {
      if (mounted) showSnack(context, 'Error: $e', error: true);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _addSkill() {
    final s = _skillCtrl.text.trim();
    if (s.isNotEmpty && !_skills.contains(s)) {
      setState(() { _skills.add(s); _skillCtrl.clear(); });
    }
  }

  @override
  void dispose() {
    _nameCtrl.dispose(); _bioCtrl.dispose(); _majorCtrl.dispose();
    _yearCtrl.dispose(); _skillCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Edit Profile')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(children: [
          GestureDetector(
            onTap: _pickImage,
            child: Stack(children: [
              _imageFile != null
                  ? CircleAvatar(radius: 52, backgroundImage: FileImage(_imageFile!))
                  : buildAvatar(widget.data['photoUrl'] as String?, r: 52),
              Positioned(
                bottom: 0, right: 0,
                child: Container(
                  padding: const EdgeInsets.all(6),
                  decoration: const BoxDecoration(color: kPrimary, shape: BoxShape.circle),
                  child: const Icon(Icons.camera_alt, size: 16, color: Colors.white),
                ),
              ),
            ]),
          ),
          const SizedBox(height: 20),
          _field('Full Name', _nameCtrl, Icons.person_outlined),
          const SizedBox(height: 14),
          _field('Bio', _bioCtrl, Icons.info_outline, lines: 3),
          const SizedBox(height: 14),
          _field('Major / Department', _majorCtrl, Icons.school_outlined),
          const SizedBox(height: 14),
          _field('Year (e.g. 2nd Year)', _yearCtrl, Icons.calendar_today),
          const SizedBox(height: 14),
          const Align(alignment: Alignment.centerLeft, child: Text('Skills', style: TextStyle(fontWeight: FontWeight.w600))),
          const SizedBox(height: 6),
          Row(children: [
            Expanded(child: TextFormField(controller: _skillCtrl, decoration: const InputDecoration(hintText: 'Add skill...'))),
            const SizedBox(width: 8),
            IconButton(onPressed: _addSkill, icon: const Icon(Icons.add_circle, color: kPrimary)),
          ]),
          const SizedBox(height: 8),
          Wrap(spacing: 8, children: _skills.map((s) => Chip(
            label: Text(s), deleteIcon: const Icon(Icons.close, size: 14),
            onDeleted: () => setState(() => _skills.remove(s)),
            backgroundColor: kPrimary.withOpacity(0.1),
            labelStyle: const TextStyle(color: kPrimary),
          )).toList()),
          const SizedBox(height: 24),
          _loading ? buildLoading() : buildGradientButton(label: 'Save Changes', onTap: _save, icon: Icons.save),
        ]),
      ),
    );
  }

  Widget _field(String label, TextEditingController ctrl, IconData icon, {int lines = 1}) => TextFormField(
    controller: ctrl, maxLines: lines,
    decoration: InputDecoration(labelText: label, prefixIcon: lines == 1 ? Icon(icon, color: kPrimary) : null),
  );
}

// ============================================================
// PORTFOLIO SCREEN
// ============================================================
class PortfolioScreen extends StatefulWidget {
  const PortfolioScreen({super.key});
  @override
  State<PortfolioScreen> createState() => _PortfolioScreenState();
}

class _PortfolioScreenState extends State<PortfolioScreen> {
  final _titleCtrl = TextEditingController();
  final _descCtrl  = TextEditingController();
  bool _adding = false;

  Future<void> _add() async {
    if (_titleCtrl.text.trim().isEmpty) return;
    await _firestore.collection('users').doc(_auth.currentUser!.uid).collection('portfolio').add({
      'title': _titleCtrl.text.trim(),
      'description': _descCtrl.text.trim(),
      'createdAt': FieldValue.serverTimestamp(),
    });
    _titleCtrl.clear(); _descCtrl.clear();
    setState(() => _adding = false);
  }

  @override
  void dispose() { _titleCtrl.dispose(); _descCtrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    final uid = _auth.currentUser?.uid ?? '';
    return Scaffold(
      appBar: AppBar(
        title: const Text('Portfolio'),
        actions: [IconButton(icon: const Icon(Icons.add), onPressed: () => setState(() => _adding = !_adding))],
      ),
      body: Column(children: [
        if (_adding)
          Padding(
            padding: const EdgeInsets.all(16),
            child: Card(child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(children: [
                TextFormField(controller: _titleCtrl, decoration: const InputDecoration(labelText: 'Project Title')),
                const SizedBox(height: 10),
                TextFormField(controller: _descCtrl, maxLines: 3, decoration: const InputDecoration(labelText: 'Description')),
                const SizedBox(height: 12),
                ElevatedButton(onPressed: _add, child: const Text('Add Project')),
              ]),
            )),
          ),
        Expanded(
          child: StreamBuilder<QuerySnapshot>(
            stream: _firestore.collection('users').doc(uid).collection('portfolio').orderBy('createdAt', descending: true).snapshots(),
            builder: (ctx, snap) {
              if (snap.connectionState == ConnectionState.waiting) return buildLoading();
              final docs = snap.data?.docs ?? [];
              if (docs.isEmpty) return buildEmpty('No projects yet. Showcase your work!', icon: Icons.work_outline);
              return ListView.builder(
                itemCount: docs.length,
                itemBuilder: (_, i) {
                  final d = docs[i].data() as Map<String, dynamic>;
                  return buildCardTile(
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Row(children: [
                        const Icon(Icons.code, color: kPrimary),
                        const SizedBox(width: 10),
                        Expanded(child: Text(d['title'] ?? '', style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15))),
                        IconButton(
                          icon: const Icon(Icons.delete_outline, color: kError, size: 18),
                          onPressed: () => docs[i].reference.delete(),
                        ),
                      ]),
                      if ((d['description'] as String? ?? '').isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(top: 6),
                          child: Text(d['description'] ?? '', style: const TextStyle(color: kTextSub)),
                        ),
                    ]),
                  );
                },
              );
            },
          ),
        ),
      ]),
    );
  }
}

// ============================================================
// SETTINGS SCREEN
// ============================================================
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        children: [
          buildSectionHeader('Account'),
          _settingsTile(context, Icons.person_outlined, 'Edit Profile', () => Navigator.push(context, _route(EditProfileScreen(data: const {})))),
          _settingsTile(context, Icons.lock_outlined, 'Change Password', () => Navigator.push(context, _route(const ForgotPasswordScreen()))),
          buildSectionHeader('Content'),
          _settingsTile(context, Icons.category_outlined, 'Categories', () => Navigator.push(context, _route(const CategoriesScreen()))),
          _settingsTile(context, Icons.groups_outlined, 'My Teams', () => Navigator.push(context, _route(const AcceptedTeamsScreen()))),
          _settingsTile(context, Icons.work_outline, 'Portfolio', () => Navigator.push(context, _route(const PortfolioScreen()))),
          buildSectionHeader('Support'),
          _settingsTile(context, Icons.info_outlined, 'About App', () => Navigator.push(context, _route(const AboutAppScreen()))),
          _settingsTile(context, Icons.help_outline, 'Help & Support', () => Navigator.push(context, _route(const HelpSupportScreen()))),
          buildSectionHeader('Danger Zone'),
          _settingsTile(context, Icons.logout, 'Sign Out', () async {
            await signOut();
            if (context.mounted) Navigator.pushAndRemoveUntil(context, _route(const LoginScreen()), (_) => false);
          }, color: kError),
        ],
      ),
    );
  }

  Widget _settingsTile(BuildContext ctx, IconData icon, String label, VoidCallback onTap, {Color? color}) => ListTile(
    leading: Icon(icon, color: color ?? kPrimary),
    title: Text(label, style: TextStyle(color: color ?? kText, fontWeight: FontWeight.w500)),
    trailing: Icon(Icons.chevron_right, color: color ?? kTextSub),
    onTap: onTap,
  );
}

// ============================================================
// ABOUT APP SCREEN
// ============================================================
class AboutAppScreen extends StatelessWidget {
  const AboutAppScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('About App')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: kPrimary.withOpacity(0.1), shape: BoxShape.circle),
            child: const Icon(Icons.groups_2, size: 64, color: kPrimary),
          ),
          const SizedBox(height: 16),
          const Text('JIHC TeamUp', style: TextStyle(fontSize: 26, fontWeight: FontWeight.w800, color: kText)),
          const Text('Version 1.0.0', style: TextStyle(color: kTextSub)),
          const SizedBox(height: 24),
          _aboutCard('What is JIHC TeamUp?',
              'JIHC TeamUp is a collaborative platform for JIHC students to find teammates, build projects, and grow their skills together.'),
          const SizedBox(height: 12),
          _aboutCard('Developer',
              'Kausar Oralbek\nStudent ID: 080626652754\nJIHC — Jere Intuitional Higher College'),
          const SizedBox(height: 12),
          _aboutCard('Technologies',
              'Flutter • Firebase Auth • Cloud Firestore\nFirebase Storage • Google Sign-In'),
          const SizedBox(height: 12),
          _aboutCard('Features',
              '• Post and discover team opportunities\n• Real-time messaging\n• Join request system\n• Portfolio showcase\n• Category-based filtering'),
        ]),
      ),
    );
  }

  Widget _aboutCard(String title, String body) => Card(
    child: Padding(
      padding: const EdgeInsets.all(16),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(title, style: const TextStyle(fontWeight: FontWeight.w700, color: kPrimary)),
        const SizedBox(height: 8),
        Text(body, style: const TextStyle(color: kTextSub, height: 1.5)),
      ]),
    ),
  );
}

// ============================================================
// HELP & SUPPORT SCREEN
// ============================================================
class HelpSupportScreen extends StatelessWidget {
  const HelpSupportScreen({super.key});

  static const _faqs = [
    ('How do I create a team post?', 'Tap the + button on the Teams tab or use the quick action on Home screen.'),
    ('How do I join a team?', 'Open a post and tap "Request to Join". The team owner will accept or decline.'),
    ('How does messaging work?', 'Open a post, tap "Message Author" to start a direct chat.'),
    ('Can I edit my post?', 'Yes, tap the edit icon on your own posts.'),
    ('How do I update my profile?', 'Go to Profile tab and tap the edit icon.'),
    ('Why is real-time not working?', 'Ensure you have an active internet connection. Firebase requires network access.'),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Help & Support')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(gradient: const LinearGradient(colors: [kPrimary, kPrimaryL]), borderRadius: BorderRadius.circular(16)),
            child: const Column(children: [
              Icon(Icons.support_agent, size: 48, color: Colors.white),
              SizedBox(height: 10),
              Text('How can we help?', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w700)),
              Text('Browse frequently asked questions below.', style: TextStyle(color: Colors.white70, fontSize: 13)),
            ]),
          ),
          const SizedBox(height: 16),
          buildSectionHeader('FAQ'),
          ..._faqs.map((faq) => Card(
            margin: const EdgeInsets.only(bottom: 8),
            child: ExpansionTile(
              title: Text(faq.$1, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
              children: [Padding(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                child: Text(faq.$2, style: const TextStyle(color: kTextSub, height: 1.5)),
              )],
            ),
          )),
          const SizedBox(height: 12),
          buildSectionHeader('Contact'),
          buildCardTile(child: const Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text('Email Support', style: TextStyle(fontWeight: FontWeight.w700)),
            SizedBox(height: 4),
            Text('daurenoralbek2402@gmail.com', style: TextStyle(color: kPrimary)),
          ])),
        ],
      ),
    );
  }
}

// ============================================================
// HELPER: page transition
// ============================================================
PageRoute _route(Widget page) => MaterialPageRoute(builder: (_) => page);

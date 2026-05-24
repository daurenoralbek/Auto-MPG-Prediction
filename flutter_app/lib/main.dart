import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
    ),
  );
  runApp(const YandexBooksApp());
}

// ══════════════════════════════════════════════════════════
// FIRESTORE (local simulation)
// ══════════════════════════════════════════════════════════
class FS {
  static const uid = 'kausar_b8305831805';
  static final _userController = StreamController<UserStats>.broadcast();
  static int _minToday = 0;

  static Future<void> saveProgress(String bookId, int pct) async {
    final idx = kBooks.indexWhere((bk) => bk.id == bookId);
    if (idx != -1) kBooks[idx].readPercent = pct;
  }

  static Future<void> saveBookmark(String bookId, bool val) async {
    final idx = kBooks.indexWhere((bk) => bk.id == bookId);
    if (idx != -1) kBooks[idx].isBookmarked = val;
  }

  static Future<void> loadAll() async {
    _emitStats();
  }

  static Stream<UserStats> userStream() async* {
    yield UserStats(minToday: _minToday);
    yield* _userController.stream;
  }

  static Future<void> updateStats(int minToday) async {
    _minToday = minToday;
    _emitStats();
  }

  static void _emitStats() {
    if (!_userController.isClosed) {
      _userController.add(UserStats(minToday: _minToday));
    }
  }
}

class UserStats {
  final int minToday;
  const UserStats({required this.minToday});
}

// ══════════════════════════════════════════════════════════
// COLORS
// ══════════════════════════════════════════════════════════
class C {
  static const bg = Color(0xFFFFFFFF);
  static const bgGray = Color(0xFFF2F2F7);
  static const border = Color(0xFFE5E5EA);
  static const txt = Color(0xFF000000);
  static const txtGray = Color(0xFF8E8E93);
  static const txtLight = Color(0xFFAEAEB2);
  static const btnDark = Color(0xFF2C2C2E);
  static const peach = Color(0xFFFFF1E6);
  static const pinkGlow = Color(0xFFFFE4EC);
  static const red = Color(0xFFFF3B30);
  static const green = Color(0xFF34C759);
  static const g1 = Color(0xFFE040FB);
  static const g2 = Color(0xFFFF6B9D);
  static const g3 = Color(0xFFFFB347);
}

// ══════════════════════════════════════════════════════════
// MODEL
// ══════════════════════════════════════════════════════════
class Book {
  final String id, title, author, genre, description;
  final double rating;
  final int reviews, pages;
  final Color coverColor;
  final String coverEmoji;
  final bool isFree, isNew, isAudio;
  final double price;
  int readPercent;
  bool isBookmarked;

  Book({
    required this.id,
    required this.title,
    required this.author,
    required this.genre,
    required this.description,
    required this.rating,
    required this.reviews,
    required this.pages,
    required this.coverColor,
    required this.coverEmoji,
    this.isFree = false,
    this.isNew = false,
    this.isAudio = false,
    this.price = 0,
    this.readPercent = 0,
    this.isBookmarked = false,
  });
}

final List<Book> kBooks = [
  Book(
    id: '1',
    title: 'Ведьмин Капучино\nи тайна наследства',
    author: 'Елена Михалёва',
    genre: 'Фэнтези',
    description:
        'Захватывающий роман о ведьме, кофе и тайнах старинного особняка.',
    rating: 4.7,
    reviews: 8420,
    pages: 320,
    coverColor: const Color(0xFFE8C5A0),
    coverEmoji: '🧙',
    isNew: true,
    price: 349,
  ),
  Book(
    id: '2',
    title: 'Война Миров',
    author: 'Герберт Уэллс',
    genre: 'Классика',
    description: 'Марсиане нападают на Землю. Классика научной фантастики.',
    rating: 4.8,
    reviews: 15200,
    pages: 280,
    coverColor: const Color(0xFF1A2740),
    coverEmoji: '🚀',
    isFree: true,
    isNew: true,
  ),
  Book(
    id: '3',
    title: '48 законов власти',
    author: 'Роберт Грин',
    genre: 'Саморазвитие',
    description: '48 законов, которые помогут вам обрести власть и удержать её.',
    rating: 4.6,
    reviews: 32100,
    pages: 496,
    coverColor: const Color(0xFFF5F0E8),
    coverEmoji: '♟️',
    price: 499,
  ),
  Book(
    id: '4',
    title: 'Я иду искать',
    author: 'Ева Меркачёва',
    genre: 'Документальная',
    description: 'Подлинные истории о российских маньяках. 18+.',
    rating: 4.5,
    reviews: 6800,
    pages: 368,
    coverColor: const Color(0xFF1A1A1A),
    coverEmoji: '🔍',
    price: 399,
    isNew: true,
  ),
  Book(
    id: '5',
    title: 'Дочери колыбели',
    author: 'Александра Яковлева',
    genre: 'Романтика',
    description: 'История о любви, потерях и возрождении.',
    rating: 4.4,
    reviews: 4200,
    pages: 288,
    coverColor: const Color(0xFF2C1A3A),
    coverEmoji: '🌸',
    price: 329,
    isNew: true,
  ),
  Book(
    id: '6',
    title: 'К себе нежно',
    author: 'Ольга Примаченко',
    genre: 'Психология',
    description: 'Книга о том, как ценить и беречь себя.',
    rating: 4.9,
    reviews: 41000,
    pages: 240,
    coverColor: const Color(0xFFFFB6C1),
    coverEmoji: '💗',
    price: 279,
  ),
  Book(
    id: '7',
    title: 'Человек государев',
    author: 'Александр Горбов',
    genre: 'Историческая',
    description: 'Аудиороман об эпохе Петра Великого.',
    rating: 4.7,
    reviews: 9100,
    pages: 512,
    coverColor: const Color(0xFF1A2C1A),
    coverEmoji: '⚔️',
    isAudio: true,
    price: 449,
  ),
  Book(
    id: '8',
    title: 'Займись ничем',
    author: 'Джозеф Джебелли',
    genre: 'Саморазвитие',
    description: 'Система долгосрочной продуктивности от нейробиолога.',
    rating: 4.5,
    reviews: 7600,
    pages: 304,
    coverColor: const Color(0xFF2A4A3A),
    coverEmoji: '🌿',
    isAudio: true,
    price: 399,
  ),
  Book(
    id: '9',
    title: 'Мастер и Маргарита',
    author: 'Михаил Булгаков',
    genre: 'Классика',
    description: 'Когда дьявол пришёл в Москву.',
    rating: 4.9,
    reviews: 52000,
    pages: 480,
    coverColor: const Color(0xFF1A1035),
    coverEmoji: '🌙',
    isFree: true,
    readPercent: 45,
    isAudio: true,
  ),
  Book(
    id: '10',
    title: '1984',
    author: 'Джордж Оруэлл',
    genre: 'Антиутопия',
    description: 'Большой Брат следит за тобой.',
    rating: 4.9,
    reviews: 38760,
    pages: 368,
    coverColor: const Color(0xFF1C1C1C),
    coverEmoji: '👁️',
    isFree: true,
    readPercent: 100,
  ),
  Book(
    id: '11',
    title: 'Дюна',
    author: 'Фрэнк Герберт',
    genre: 'Фантастика',
    description: 'Эпическая сага о пустынной планете Арракис.',
    rating: 4.9,
    reviews: 31204,
    pages: 896,
    coverColor: const Color(0xFF5C3010),
    coverEmoji: '🏜️',
    price: 399,
    readPercent: 22,
    isNew: true,
    isAudio: true,
  ),
  Book(
    id: '12',
    title: 'Атомные привычки',
    author: 'Джеймс Клир',
    genre: 'Саморазвитие',
    description: 'Небольшие изменения — выдающиеся результаты.',
    rating: 4.8,
    reviews: 42100,
    pages: 320,
    coverColor: const Color(0xFF1B3A1B),
    coverEmoji: '⚡',
    price: 499,
    isNew: true,
    isAudio: true,
  ),
  Book(
    id: '13',
    title: 'Прачечная, стирающая печали',
    author: 'Ким Чжи Юн',
    genre: 'Романтика',
    description: 'Волшебная прачечная, которая стирает печали.',
    rating: 4.6,
    reviews: 12000,
    pages: 192,
    coverColor: const Color(0xFF3A5A8A),
    coverEmoji: '🧺',
    isAudio: true,
    price: 259,
  ),
  Book(
    id: '14',
    title: 'Последнее дело майора Чистова',
    author: 'Евгений Водолазкин',
    genre: 'Детективы',
    description: 'Детектив о майоре Чистове и его последнем деле.',
    rating: 4.7,
    reviews: 5400,
    pages: 336,
    coverColor: const Color(0xFF2A2A2A),
    coverEmoji: '🔎',
    isNew: true,
    price: 369,
  ),
  Book(
    id: '15',
    title: 'Маленький принц',
    author: 'Антуан де Сент-Экзюпери',
    genre: 'Классика',
    description: 'Философская сказка для взрослых.',
    rating: 4.9,
    reviews: 52000,
    pages: 112,
    coverColor: const Color(0xFF0D1F3C),
    coverEmoji: '⭐',
    isFree: true,
    readPercent: 100,
    isAudio: true,
  ),
];

const kCategories = [
  'Проза',
  'Классика',
  'Саморазвитие',
  'Психология',
  'Фэнтези',
  'Романтика',
  'Здоровье',
  'Бесплатно',
  'Фантастика',
  'Young Adult',
  'Триллеры и хорроры',
  'Нон-фикшн',
  'Детективы',
  'Бизнес',
  'Биографии и мемуары',
  'История',
  'Книги на казахском языке',
];

const kTrending = [
  'Евгений Водолазкин. Последнее дело майора Чистова',
  'Увлечь ребенка в дороге. Специально для Мосгортранс',
  'Бесплатные книги',
  'Виктор Пелевин. Возвращение Синей Бороды',
  'Только в Яндекс Книгах',
];

// ── helpers ────────────────────────────────────────────────
PageRoute slide(Widget p) => PageRouteBuilder(
      pageBuilder: (_, __, ___) => p,
      transitionsBuilder: (_, a, __, c) => SlideTransition(
        position: Tween(
          begin: const Offset(1, 0),
          end: Offset.zero,
        ).animate(CurvedAnimation(parent: a, curve: Curves.easeOutCubic)),
        child: c,
      ),
      transitionDuration: const Duration(milliseconds: 250),
    );

String fmt(int n) => n >= 1000 ? '${(n / 1000).toStringAsFixed(0)}K' : '$n';
PageRoute buildRoute(Widget p) => slide(p);
String fmtNum(int n) => fmt(n);

// ══════════════════════════════════════════════════════════
// APP
// ══════════════════════════════════════════════════════════
class YandexBooksApp extends StatefulWidget {
  const YandexBooksApp({super.key});

  @override
  State<YandexBooksApp> createState() => _YandexBooksAppState();
}

class _YandexBooksAppState extends State<YandexBooksApp> {
  bool _ready = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    await FS.loadAll();
    setState(() => _ready = true);
  }

  @override
  Widget build(BuildContext context) {
    if (!_ready) {
      return const MaterialApp(
        debugShowCheckedModeBanner: false,
        home: Scaffold(
          backgroundColor: Colors.white,
          body: Center(child: CircularProgressIndicator(color: Colors.black)),
        ),
      );
    }
    return MaterialApp(
      title: 'Яндекс Книги',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.black),
        useMaterial3: true,
        scaffoldBackgroundColor: C.bg,
        appBarTheme: const AppBarTheme(
          backgroundColor: C.bg,
          elevation: 0,
          scrolledUnderElevation: 0,
          foregroundColor: C.txt,
          titleTextStyle: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.w800,
            color: C.txt,
            letterSpacing: -0.5,
          ),
        ),
      ),
      home: const MainShell(),
    );
  }
}

// ══════════════════════════════════════════════════════════
// SHELL
// ══════════════════════════════════════════════════════════
class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> {
  int _tab = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _tab == 2
          ? const AIScreen()
          : IndexedStack(
              index: _tab > 2 ? _tab - 1 : _tab,
              children: const [
                MyBooksScreen(),
                FeedScreen(),
                SearchScreen(),
                ProfileScreen(),
              ],
            ),
      bottomNavigationBar: _BottomNav(
        current: _tab,
        onTap: (i) => setState(() => _tab = i),
      ),
    );
  }
}

class _BottomNav extends StatelessWidget {
  final int current;
  final void Function(int) onTap;

  const _BottomNav({required this.current, required this.onTap});

  @override
  Widget build(BuildContext context) => Container(
        decoration: BoxDecoration(
          color: Colors.white,
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.06),
              blurRadius: 8,
              offset: const Offset(0, -1),
            ),
          ],
        ),
        child: SafeArea(
          top: false,
          child: SizedBox(
            height: 50,
            child: Row(
              children: [
                _NavIco(
                  active: current == 0,
                  onTap: () => onTap(0),
                  a: Icons.menu_book,
                  i: Icons.menu_book_outlined,
                ),
                _NavIco(
                  active: current == 1,
                  onTap: () => onTap(1),
                  a: Icons.auto_stories,
                  i: Icons.auto_stories_outlined,
                ),
                // center sparkle button
                Expanded(
                  child: GestureDetector(
                    onTap: () => onTap(2),
                    behavior: HitTestBehavior.opaque,
                    child: Center(
                      child: Container(
                        width: 46,
                        height: 46,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          gradient: const LinearGradient(
                            colors: [C.g1, C.g2, C.g3],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: C.g2.withValues(alpha: 0.45),
                              blurRadius: 14,
                              spreadRadius: 1,
                            ),
                          ],
                        ),
                        child: const Stack(
                          alignment: Alignment.center,
                          children: [
                            Positioned(
                              left: 14,
                              top: 14,
                              child: Icon(
                                Icons.auto_awesome,
                                color: Colors.white,
                                size: 11,
                              ),
                            ),
                            Positioned(
                              right: 13,
                              bottom: 13,
                              child: Icon(
                                Icons.auto_awesome,
                                color: Colors.white,
                                size: 15,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
                _NavIco(
                  active: current == 3,
                  onTap: () => onTap(3),
                  a: Icons.search,
                  i: Icons.search,
                ),
                // profile — graduation cap avatar
                Expanded(
                  child: GestureDetector(
                    onTap: () => onTap(4),
                    behavior: HitTestBehavior.opaque,
                    child: Center(
                      child: Container(
                        width: 28,
                        height: 28,
                        decoration: BoxDecoration(
                          color: const Color(0xFFFFF9C4),
                          shape: BoxShape.circle,
                          border: Border.all(
                            color:
                                current == 4 ? C.txt : Colors.transparent,
                            width: 1.5,
                          ),
                        ),
                        child: Icon(
                          Icons.school,
                          size: 16,
                          color: current == 4 ? C.txt : C.txtGray,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      );
}

class _NavIco extends StatelessWidget {
  final bool active;
  final VoidCallback onTap;
  final IconData a, i;

  const _NavIco({
    required this.active,
    required this.onTap,
    required this.a,
    required this.i,
  });

  @override
  Widget build(BuildContext context) => Expanded(
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTap: onTap,
          child: Icon(active ? a : i, size: 26, color: active ? C.txt : C.txtGray),
        ),
      );
}

// ══════════════════════════════════════════════════════════
// MY BOOKS SCREEN
// ══════════════════════════════════════════════════════════
class MyBooksScreen extends StatelessWidget {
  const MyBooksScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final reading = kBooks
        .where((b) => b.readPercent > 0 && b.readPercent < 100)
        .toList();

    return Scaffold(
      backgroundColor: C.bg,
      appBar: AppBar(
        title: const Text('Мои книги'),
        centerTitle: false,
        actions: [
          IconButton(
            icon: const Icon(Icons.notifications_none_outlined, size: 26),
            onPressed: () {},
          ),
          const SizedBox(width: 4),
        ],
      ),
      body: ListView(
        children: [
          // Stats card with local stream
          StreamBuilder<UserStats>(
            stream: FS.userStream(),
            builder: (ctx, snap) {
              final minToday = snap.data?.minToday ?? 0;
              return Container(
                margin: const EdgeInsets.fromLTRB(16, 4, 16, 0),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.06),
                      blurRadius: 16,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        _SI('$minToday мин', 'сегодня'),
                        const SizedBox(width: 32),
                        _SI('0 дней', 'в мае'),
                        const SizedBox(width: 32),
                        _SI('0 дней', 'подряд'),
                        const Spacer(),
                        const Icon(Icons.chevron_right, color: C.txtGray),
                      ],
                    ),
                    const SizedBox(height: 12),
                    const Divider(color: C.border, height: 1),
                    const SizedBox(height: 12),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Container(
                          width: 40,
                          height: 40,
                          decoration: BoxDecoration(
                            gradient: const LinearGradient(
                              colors: [Color(0xFF9C27B0), Color(0xFFE040FB)],
                            ),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: const Icon(
                            Icons.view_in_ar_outlined,
                            color: Colors.white,
                            size: 22,
                          ),
                        ),
                        const SizedBox(width: 12),
                        const Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Первая награда',
                                style: TextStyle(
                                  fontWeight: FontWeight.w700,
                                  fontSize: 14,
                                ),
                              ),
                              Text(
                                'Почитайте 10 минут',
                                style: TextStyle(
                                  color: C.txtGray,
                                  fontSize: 13,
                                ),
                              ),
                            ],
                          ),
                        ),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            const Text(
                              'Прочитаю в 2026',
                              style: TextStyle(
                                  color: C.txtGray, fontSize: 12),
                            ),
                            const SizedBox(height: 6),
                            SizedBox(
                              width: 110,
                              height: 4,
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(4),
                                child: LinearProgressIndicator(
                                  value: (minToday / 10).clamp(0.0, 1.0),
                                  backgroundColor: C.bgGray,
                                  valueColor: const AlwaysStoppedAnimation(C.txt),
                                  minHeight: 4,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Align(
                      alignment: Alignment.centerRight,
                      child: Icon(
                        Icons.info_outline,
                        size: 18,
                        color: C.txtGray.withValues(alpha: 0.6),
                      ),
                    ),
                  ],
                ),
              );
            },
          ),

          // Empty / reading
          if (reading.isEmpty) ...[
            const SizedBox(height: 16),
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 16),
              padding: const EdgeInsets.fromLTRB(24, 32, 24, 28),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(24),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.06),
                    blurRadius: 16,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Column(
                children: [
                  _OpenBookIcon(),
                  const SizedBox(height: 16),
                  const Text(
                    'Перейти в Библиотеку',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'А здесь будет всё, что вам\nзахочется почитать',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: C.txtGray, fontSize: 13),
                  ),
                  const SizedBox(height: 16),
                  GestureDetector(
                    onTap: () {},
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      decoration: BoxDecoration(
                        color: C.txt,
                        borderRadius: BorderRadius.circular(28),
                      ),
                      alignment: Alignment.center,
                      child: const Text(
                        'Найти интересные книги',
                        style: TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                          fontSize: 16,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ] else ...[
            const SizedBox(height: 16),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
              child: Row(
                children: [
                  const Text(
                    'Сейчас читаю',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700),
                  ),
                  const Spacer(),
                  Text(
                    '${reading.length}',
                    style: const TextStyle(color: C.txtGray, fontSize: 14),
                  ),
                  const Icon(Icons.chevron_right, color: C.txtGray, size: 18),
                ],
              ),
            ),
            ...reading.map(
              (b) => _ReadTile(
                book: b,
                onTap: () => Navigator.push(
                    context, slide(BookDetailScreen(book: b))),
              ),
            ),
          ],

          const SizedBox(height: 16),

          // Цитаты / Заметки / Впечатления
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            decoration: const BoxDecoration(
              border: Border(
                top: BorderSide(color: C.border),
                bottom: BorderSide(color: C.border),
              ),
            ),
            child: Row(
              children: [
                _IStat(Icons.auto_awesome_outlined, 'Цитаты', 0),
                Container(width: 1, height: 72, color: C.border),
                _IStat(Icons.edit_note_outlined, 'Заметки', 0),
                Container(width: 1, height: 72, color: C.border),
                _IStat(Icons.sentiment_satisfied_outlined, 'Впечатления', 0),
              ],
            ),
          ),

          const SizedBox(height: 16),

          // Menu rows
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            decoration: const BoxDecoration(
              border: Border(bottom: BorderSide(color: C.border)),
            ),
            child: Column(
              children: [
                _MRow(
                  icon: Icons.book_outlined,
                  label: 'Все',
                  count: kBooks.length,
                ),
                _div(),
                _MRow(
                  icon: Icons.flag_outlined,
                  label: 'Законченные',
                  count: kBooks.where((b) => b.readPercent == 100).length,
                ),
                _div(),
                _MRow(icon: Icons.shelves, label: 'Мои полки', count: 0),
                _div(),
                _MRow(
                    icon: Icons.shelves, label: 'Слежу за полками', count: 0),
                _div(),
                _MRow(
                  icon: Icons.person_outline,
                  label: 'Подписки на авторов',
                  count: 0,
                ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          // Upload EPUB
          GestureDetector(
            onTap: () {},
            child: const Padding(
              padding: EdgeInsets.symmetric(vertical: 48),
              child: Column(
                children: [
                  Icon(Icons.cloud_upload_outlined, size: 36, color: C.txtGray),
                  SizedBox(height: 12),
                  Text(
                    'Загрузить книгу в формате',
                    style: TextStyle(fontSize: 15, color: C.txtGray),
                  ),
                  Text(
                    'EPUB или FB2',
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      color: C.txt,
                    ),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 32),
        ],
      ),
    );
  }
}

Widget _div() => const Divider(height: 1, color: C.border, indent: 52);

class _SI extends StatelessWidget {
  final String v, l;

  const _SI(this.v, this.l);

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            v,
            style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 18),
          ),
          Text(l, style: const TextStyle(color: C.txtGray, fontSize: 11)),
        ],
      );
}

class _IStat extends StatelessWidget {
  final IconData icon;
  final String label;
  final int count;

  const _IStat(this.icon, this.label, this.count);

  @override
  Widget build(BuildContext context) => Expanded(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 16),
          child: Column(
            children: [
              Icon(icon, size: 24, color: C.txt),
              const SizedBox(height: 8),
              Text(
                label,
                style: const TextStyle(
                  fontSize: 13,
                  color: C.txt,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                '$count',
                style: const TextStyle(fontSize: 14, color: C.txtGray),
              ),
            ],
          ),
        ),
      );
}

class _MRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final int count;

  const _MRow({required this.icon, required this.label, required this.count});

  @override
  Widget build(BuildContext context) => ListTile(
        dense: true,
        leading: Icon(icon, size: 20, color: C.txt),
        title: Text(label, style: const TextStyle(fontSize: 15, color: C.txt)),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('$count',
                style: const TextStyle(color: C.txtGray, fontSize: 14)),
            const SizedBox(width: 4),
            const Icon(Icons.chevron_right, color: C.txtGray, size: 18),
          ],
        ),
        onTap: () {},
      );
}

class _ReadTile extends StatelessWidget {
  final Book book;
  final VoidCallback onTap;

  const _ReadTile({required this.book, required this.onTap});

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: onTap,
        child: Container(
          margin: const EdgeInsets.fromLTRB(16, 0, 16, 8),
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: C.border),
          ),
          child: Row(
            children: [
              BookCover(book: book, width: 52, height: 72),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      book.title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      book.author,
                      style: const TextStyle(color: C.txtGray, fontSize: 12),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          '${book.readPercent}%',
                          style: const TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 12,
                          ),
                        ),
                        Text(
                          '${100 - book.readPercent}% осталось',
                          style: const TextStyle(
                              color: C.txtGray, fontSize: 11),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: book.readPercent / 100,
                        backgroundColor: C.bgGray,
                        valueColor: const AlwaysStoppedAnimation(C.txt),
                        minHeight: 4,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      );
}

class _OpenBookIcon extends StatelessWidget {
  @override
  Widget build(BuildContext context) => Container(
        width: 88,
        height: 88,
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFFFF8A65), Color(0xFFFF5252)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(20),
        ),
        child: const Icon(Icons.menu_book, color: Colors.white, size: 48),
      );
}

// ══════════════════════════════════════════════════════════
// FEED SCREEN
// ══════════════════════════════════════════════════════════
class FeedScreen extends StatefulWidget {
  const FeedScreen({super.key});

  @override
  State<FeedScreen> createState() => _FeedScreenState();
}

class _FeedScreenState extends State<FeedScreen>
    with SingleTickerProviderStateMixin {
  late TabController tc;

  @override
  void initState() {
    super.initState();
    tc = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    tc.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final newBooks = kBooks.where((b) => b.isNew).toList();
    final audioNew = kBooks.where((b) => b.isAudio && b.isNew).toList();
    final popular = List<Book>.from(kBooks)
      ..sort((a, b) => b.reviews.compareTo(a.reviews));

    return Scaffold(
      backgroundColor: C.bg,
      body: NestedScrollView(
        headerSliverBuilder: (ctx, _) => [
          SliverAppBar(
            floating: true,
            snap: true,
            elevation: 0,
            scrolledUnderElevation: 0,
            backgroundColor: C.bg,
            toolbarHeight: 52,
            titleSpacing: 16,
            title: Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                _TB('Главное', 0, tc),
                const SizedBox(width: 20),
                _TB('Аудио', 1, tc),
                const SizedBox(width: 20),
                _TB('Детям', 2, tc),
              ],
            ),
          ),
        ],
        body: TabBarView(
          controller: tc,
          children: [
            _MainFeed(
                newBooks: newBooks, audioNew: audioNew, popular: popular),
            _AudioFeed(
                audioBooks: kBooks.where((b) => b.isAudio).toList()),
            const _KidsFeed(),
          ],
        ),
      ),
    );
  }
}

class _TB extends StatelessWidget {
  final String label;
  final int index;
  final TabController tc;

  const _TB(this.label, this.index, this.tc);

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
        animation: tc,
        builder: (_, __) {
          final active = tc.index == index;
          return GestureDetector(
            onTap: () => tc.animateTo(index),
            child: Text(
              label,
              style: TextStyle(
                fontSize: active ? 28 : 22,
                fontWeight:
                    active ? FontWeight.w800 : FontWeight.w500,
                color: active ? C.txt : C.txtGray,
                letterSpacing: -0.5,
                height: 1.1,
              ),
            ),
          );
        },
      );
}

class _MainFeed extends StatelessWidget {
  final List<Book> newBooks, audioNew, popular;

  const _MainFeed({
    required this.newBooks,
    required this.audioNew,
    required this.popular,
  });

  @override
  Widget build(BuildContext context) => ListView(
        children: [
          const _ChipsRow(labels: ['Проза', 'Классика', 'Саморазвитие']),
          _HeroCarousel(
            books: popular.take(5).toList(),
            addLabel: 'Добавить книгу',
          ),
          const SizedBox(height: 8),
          _SH(title: 'Вам может понравиться', action: 'Все', onAction: () {}),
          const SizedBox(height: 12),
          SizedBox(
              height: 168,
              child: _CoversRow(books: popular.take(6).toList())),
          const SizedBox(height: 28),
          _SH(title: 'Новинки', action: 'Все', onAction: () {}),
          const SizedBox(height: 12),
          SizedBox(height: 168, child: _CoversRow(books: newBooks)),
          const SizedBox(height: 28),
          _SH(title: 'Новинки: аудио', action: 'Все', onAction: () {}),
          const SizedBox(height: 12),
          SizedBox(height: 168, child: _AudioRow(books: audioNew)),
          const SizedBox(height: 28),
          _SH(
              title: 'Ещё больше отличных книг',
              action: 'Все',
              onAction: () {}),
          const SizedBox(height: 12),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: AspectRatio(
                aspectRatio: 2.2,
                child: Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        const Color(0xFF8B7355).withValues(alpha: 0.85),
                        const Color(0xFF4A4A4A).withValues(alpha: 0.9),
                      ],
                    ),
                  ),
                  child: Stack(
                    children: [
                      ...List.generate(4, (i) {
                        final b = popular[i + 2];
                        return Positioned(
                          left: 20.0 + i * 55,
                          top: 8,
                          bottom: 8,
                          child: BookCover(
                            book: b,
                            width: 70,
                            height: double.infinity,
                          ),
                        );
                      }),
                    ],
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 32),
        ],
      );
}

class _AudioFeed extends StatelessWidget {
  final List<Book> audioBooks;

  const _AudioFeed({required this.audioBooks});

  @override
  Widget build(BuildContext context) => ListView(
        children: [
          const _ChipsRow(labels: ['Проза', 'Классика', 'Саморазвитие']),
          _HeroCarousel(
            books: audioBooks.take(5).toList(),
            addLabel: 'Добавить аудиокнигу',
            showToggle: true,
          ),
          const SizedBox(height: 8),
          _SH(title: 'Суперхиты', action: 'Все', onAction: () {}),
          const SizedBox(height: 12),
          SizedBox(
            height: 220,
            child: _SuperhitsRow(books: audioBooks.take(4).toList()),
          ),
          const SizedBox(height: 32),
        ],
      );
}

class _KidsFeed extends StatelessWidget {
  const _KidsFeed();

  @override
  Widget build(BuildContext context) {
    final kids = kBooks.take(6).toList();
    return ListView(
      children: [
        const _ChipsRow(
            labels: ['13-16 лет', 'Малышам', '4-6 лет', 'Аудио']),
        _HeroCarousel(books: kids, addLabel: 'Добавить книгу'),
        const SizedBox(height: 8),
        _SH(title: 'Вам может понравиться', action: 'Все', onAction: () {}),
        const SizedBox(height: 12),
        SizedBox(height: 168, child: _CoversRow(books: kids)),
        const SizedBox(height: 32),
      ],
    );
  }
}

// ══════════════════════════════════════════════════════════
// SEARCH SCREEN
// ══════════════════════════════════════════════════════════
class SearchScreen extends StatefulWidget {
  const SearchScreen({super.key});

  @override
  State<SearchScreen> createState() => _SearchScreenState();
}

class _SearchScreenState extends State<SearchScreen> {
  final ctrl = TextEditingController();
  final focus = FocusNode();
  List<Book> results = [];
  bool hasQ = false;
  bool focused = false;

  @override
  void initState() {
    super.initState();
    focus.addListener(() => setState(() => focused = focus.hasFocus));
  }

  @override
  void dispose() {
    ctrl.dispose();
    focus.dispose();
    super.dispose();
  }

  void _search(String q) {
    final ql = q.toLowerCase().trim();
    setState(() {
      hasQ = q.isNotEmpty;
      results = ql.isEmpty
          ? []
          : kBooks
              .where(
                (b) =>
                    b.title.toLowerCase().contains(ql) ||
                    b.author.toLowerCase().contains(ql) ||
                    b.genre.toLowerCase().contains(ql) ||
                    b.description.toLowerCase().contains(ql),
              )
              .toList();
    });
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        backgroundColor: C.bg,
        body: SafeArea(
          child: Column(
            children: [
              // search bar
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
                child: Row(
                  children: [
                    Expanded(
                      child: Container(
                        height: 40,
                        decoration: BoxDecoration(
                          color: C.bgGray,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: TextField(
                          controller: ctrl,
                          focusNode: focus,
                          onChanged: _search,
                          style: const TextStyle(fontSize: 16, color: C.txt),
                          decoration: InputDecoration(
                            hintText: 'Книги, авторы, жанры',
                            hintStyle: const TextStyle(
                              color: C.txtGray,
                              fontSize: 16,
                            ),
                            prefixIcon: const Icon(
                              Icons.search,
                              color: C.txtGray,
                              size: 20,
                            ),
                            border: InputBorder.none,
                            contentPadding:
                                const EdgeInsets.symmetric(vertical: 10),
                            suffixIcon: ctrl.text.isNotEmpty
                                ? IconButton(
                                    icon: const Icon(
                                      Icons.close,
                                      size: 18,
                                      color: C.txtGray,
                                    ),
                                    onPressed: () {
                                      ctrl.clear();
                                      _search('');
                                    },
                                  )
                                : null,
                          ),
                        ),
                      ),
                    ),
                    if (focused || hasQ) ...[
                      const SizedBox(width: 12),
                      GestureDetector(
                        onTap: () {
                          ctrl.clear();
                          _search('');
                          focus.unfocus();
                        },
                        child: const Text(
                          'Отменить',
                          style: TextStyle(fontSize: 16, color: C.txt),
                        ),
                      ),
                    ],
                  ],
                ),
              ),

              Expanded(
                child: focused && !hasQ
                    ? const _SearchFocused()
                    : !hasQ
                        ? const _SearchHome()
                        : results.isEmpty
                            ? const Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(Icons.search_off,
                                        size: 64, color: C.border),
                                    SizedBox(height: 12),
                                    Text(
                                      'Ничего не найдено',
                                      style: TextStyle(
                                        fontSize: 18,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                    SizedBox(height: 4),
                                    Text(
                                      'Попробуйте другой запрос',
                                      style: TextStyle(color: C.txtGray),
                                    ),
                                  ],
                                ),
                              )
                            : Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Padding(
                                    padding: const EdgeInsets.fromLTRB(
                                        16, 12, 16, 4),
                                    child: Text(
                                      'Найдено: ${results.length}',
                                      style: const TextStyle(
                                        color: C.txtGray,
                                        fontSize: 13,
                                      ),
                                    ),
                                  ),
                                  Expanded(
                                    child: ListView.builder(
                                      itemCount: results.length,
                                      itemBuilder: (ctx, i) => BookRow(
                                        book: results[i],
                                        onTap: () => Navigator.push(
                                          ctx,
                                          slide(BookDetailScreen(
                                              book: results[i])),
                                        ),
                                      ),
                                    ),
                                  ),
                                ],
                              ),
              ),
            ],
          ),
        ),
      );
}

class _SearchFocused extends StatelessWidget {
  const _SearchFocused();

  @override
  Widget build(BuildContext context) => Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFFFF5F6D), Color(0xFFFFC371)],
                    begin: Alignment.bottomLeft,
                    end: Alignment.topRight,
                  ),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Icon(Icons.search, color: Colors.white, size: 40),
              ),
              const SizedBox(height: 24),
              const Text(
                'Попробуйте поискать книгу или автора',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                  color: C.txt,
                  height: 1.25,
                ),
              ),
              const SizedBox(height: 10),
              const Text(
                'История поиска останется на этой странице',
                textAlign: TextAlign.center,
                style:
                    TextStyle(fontSize: 15, color: C.txtGray, height: 1.35),
              ),
            ],
          ),
        ),
      );
}

class _SearchHome extends StatelessWidget {
  const _SearchHome();

  @override
  Widget build(BuildContext context) {
    final rec = kBooks.take(5).toList();
    final audio = kBooks.where((b) => b.isAudio).take(3).toList();

    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      children: [
        const SizedBox(height: 12),
        const Text(
          'Сейчас популярно',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w800,
            color: C.txt,
          ),
        ),
        const SizedBox(height: 10),
        Text(
          kTrending.join(' · '),
          style: const TextStyle(color: C.txt, fontSize: 16, height: 1.45),
        ),
        const SizedBox(height: 28),
        const Text(
          'Категории',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w800,
            color: C.txt,
          ),
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: kCategories.map((c) => _CCh(c)).toList(),
        ),
        const SizedBox(height: 28),
        _SH(title: 'Вам может понравиться', action: 'Все', onAction: () {}),
        const SizedBox(height: 12),
        SizedBox(height: 168, child: _CoversRow(books: rec)),
        const SizedBox(height: 28),
        _SH(
          title: 'Вам может понравиться: аудио',
          action: 'Все',
          onAction: () {},
        ),
        const SizedBox(height: 12),
        SizedBox(height: 168, child: _AudioRow(books: audio)),
        const SizedBox(height: 32),
      ],
    );
  }
}

class _CCh extends StatelessWidget {
  final String label;

  const _CCh(this.label);

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: C.border),
        ),
        child: Text(
          label,
          style: const TextStyle(fontSize: 14, color: C.txt, height: 1.1),
        ),
      );
}

// ══════════════════════════════════════════════════════════
// PROFILE SCREEN
// ══════════════════════════════════════════════════════════
class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) => Scaffold(
        backgroundColor: C.bg,
        body: CustomScrollView(
          slivers: [
            SliverAppBar(
              backgroundColor: C.bg,
              pinned: false,
              floating: true,
              elevation: 0,
              title: const Text(
                'Профиль',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.w800,
                  color: C.txt,
                  letterSpacing: -0.5,
                ),
              ),
              actions: [
                IconButton(
                  icon: const Icon(Icons.edit_outlined,
                      color: C.txt, size: 22),
                  onPressed: () {},
                ),
                const SizedBox(width: 4),
              ],
            ),
            SliverToBoxAdapter(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 8),
                    child: Row(
                      children: [
                        Container(
                          width: 60,
                          height: 60,
                          decoration: const BoxDecoration(
                            color: Color(0xFFFFF9C4),
                            shape: BoxShape.circle,
                          ),
                          child: const Center(
                            child:
                                Text('🎓', style: TextStyle(fontSize: 30)),
                          ),
                        ),
                        const SizedBox(width: 14),
                        const Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'каусар оралбек',
                              style: TextStyle(
                                fontSize: 17,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            SizedBox(height: 2),
                            Text(
                              '@b8305831805',
                              style: TextStyle(
                                  color: C.txtGray, fontSize: 13),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  _PS(label: 'Детский режим'),
                  Container(
                    margin: const EdgeInsets.symmetric(horizontal: 16),
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: C.bgGray,
                      borderRadius: BorderRadius.circular(18),
                    ),
                    child: Column(
                      children: [
                        Row(
                          children: [
                            Container(
                              width: 44,
                              height: 44,
                              decoration: const BoxDecoration(
                                color: Color(0xFFFFE0B2),
                                shape: BoxShape.circle,
                              ),
                              child: const Center(
                                child: Text('🐻',
                                    style: TextStyle(fontSize: 24)),
                              ),
                            ),
                            const SizedBox(width: 12),
                            const Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'Включить',
                                    style: TextStyle(
                                      fontWeight: FontWeight.w700,
                                      fontSize: 16,
                                    ),
                                  ),
                                  Text(
                                    'Книги 16+',
                                    style: TextStyle(
                                      color: C.txtGray,
                                      fontSize: 13,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            const Icon(Icons.edit_outlined,
                                size: 18, color: C.txtGray),
                            const SizedBox(width: 8),
                            const Icon(Icons.chevron_right,
                                color: C.txtGray, size: 20),
                          ],
                        ),
                        const SizedBox(height: 10),
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: C.peach,
                            borderRadius: BorderRadius.circular(14),
                          ),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text('🦊',
                                  style: TextStyle(fontSize: 28)),
                              const SizedBox(width: 12),
                              const Expanded(
                                child: Column(
                                  crossAxisAlignment:
                                      CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      'Создайте детский аккаунт',
                                      style: TextStyle(
                                        fontWeight: FontWeight.w700,
                                        fontSize: 14,
                                      ),
                                    ),
                                    SizedBox(height: 4),
                                    Text(
                                      'Аккаунт в сервисах Яндекса\nс защитой от взрослых тем',
                                      style: TextStyle(
                                        color: C.txtGray,
                                        fontSize: 13,
                                        height: 1.3,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const Icon(Icons.close,
                                  size: 18, color: C.txtGray),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  _PS(label: 'Настройки'),
                  _SG(
                    children: [
                      _SR(label: 'Подписчики', count: 0),
                      _SR(label: 'Подписки', count: 0),
                    ],
                  ),
                  const SizedBox(height: 8),
                  _SG(
                    children: [
                      _ST(label: 'Настройки чтения'),
                      _ST(label: 'Чтение без интернета'),
                      _ST(label: 'Прослушивание без интернета'),
                    ],
                  ),
                  const SizedBox(height: 8),
                  _SG(
                    children: [
                      _ST(label: 'Конфиденциальность'),
                      _ST(label: 'Настройки уведомлений'),
                      _ST(label: 'Выбор иконки'),
                      _ST(label: 'Выбор темы'),
                    ],
                  ),
                  const SizedBox(height: 8),
                  _SG(
                    children: [
                      _ST(label: 'Условия использования'),
                      _ST(label: 'Политика конфиденциальности'),
                      _ST(label: 'Правила рекомендаций'),
                    ],
                  ),
                  const SizedBox(height: 8),
                  _SG(
                    children: [
                      _ST(label: 'Активировать промокод'),
                      _ST(label: 'Частые вопросы'),
                      _ST(label: 'Чат с поддержкой'),
                      _ST(label: 'Удалить аккаунт'),
                      _ST(label: 'Выйти', red: true),
                    ],
                  ),
                  const SizedBox(height: 16),
                  const Center(
                    child: Text(
                      'Яндекс Книги 2.69.0 (26351)',
                      style: TextStyle(color: C.txtLight, fontSize: 12),
                    ),
                  ),
                  const SizedBox(height: 32),
                ],
              ),
            ),
          ],
        ),
      );
}

class _PS extends StatelessWidget {
  final String label;

  const _PS({required this.label});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 10),
        child: Text(
          label,
          style: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w800,
            color: C.txt,
          ),
        ),
      );
}

class _SG extends StatelessWidget {
  final List<Widget> children;

  const _SG({required this.children});

  @override
  Widget build(BuildContext context) => Container(
        margin: const EdgeInsets.symmetric(horizontal: 16),
        decoration: BoxDecoration(
          color: C.bgGray,
          borderRadius: BorderRadius.circular(18),
        ),
        child: Column(
          children: List.generate(children.length * 2 - 1, (i) {
            if (i.isOdd) {
              return const Padding(
                padding: EdgeInsets.only(left: 16),
                child: Divider(height: 1, color: Color(0xFFD1D1D6)),
              );
            }
            return children[i ~/ 2];
          }),
        ),
      );
}

class _SR extends StatelessWidget {
  final String label;
  final int count;

  const _SR({required this.label, required this.count});

  @override
  Widget build(BuildContext context) => ListTile(
        dense: true,
        title: Text(label, style: const TextStyle(fontSize: 16)),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('$count',
                style: const TextStyle(color: C.txtGray, fontSize: 14)),
            const SizedBox(width: 4),
            const Icon(Icons.chevron_right, size: 18, color: C.txtGray),
          ],
        ),
        onTap: () {},
      );
}

class _ST extends StatelessWidget {
  final String label;
  final bool red;

  const _ST({required this.label, this.red = false});

  @override
  Widget build(BuildContext context) => ListTile(
        dense: true,
        title: Text(
          label,
          style: TextStyle(
            fontSize: 16,
            color: red ? C.red : C.txt,
          ),
        ),
        trailing: red
            ? null
            : const Icon(Icons.chevron_right, size: 18, color: C.txtGray),
        onTap: () {},
      );
}

// ══════════════════════════════════════════════════════════
// AI / PLUS SCREEN
// ══════════════════════════════════════════════════════════
class AIScreen extends StatelessWidget {
  const AIScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: C.bg,
      body: SafeArea(
        child: Stack(
          children: [
            Column(
              children: [
                const Padding(
                  padding: EdgeInsets.only(top: 40),
                  child: _PlusCoversMarquee(),
                ),
                const Spacer(),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: Column(
                    children: const [
                      Text(
                        'Читайте и слушайте книги\nпо подписке Яндекс Плюс',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w800,
                          color: C.txt,
                          height: 1.2,
                          letterSpacing: -0.3,
                        ),
                      ),
                      SizedBox(height: 14),
                      Text(
                        'С мультиподпиской вам доступны большая библиотека, удобная читалка и рекомендации. А ещё Кинопоиск и Яндекс Музыка',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 15,
                          color: C.txtGray,
                          height: 1.4,
                        ),
                      ),
                      SizedBox(height: 12),
                      Text(
                        'Подробнее о Плюсе',
                        style: TextStyle(
                          fontSize: 15,
                          color: C.txtGray,
                          decoration: TextDecoration.underline,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(vertical: 17),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [Color(0xFFFF5E62), Color(0xFF4E65FF)],
                      ),
                      borderRadius: BorderRadius.circular(28),
                    ),
                    alignment: Alignment.center,
                    child: const Text(
                      '2 000 KZT в месяц',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 17,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ),
              ],
            ),
            Positioned(
              top: 12,
              right: 16,
              child: GestureDetector(
                onTap: () => Navigator.of(context).maybePop(),
                child: Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.1),
                        blurRadius: 8,
                      ),
                    ],
                  ),
                  child: const Icon(Icons.close, size: 20, color: C.txt),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PlusCoversMarquee extends StatefulWidget {
  const _PlusCoversMarquee();

  @override
  State<_PlusCoversMarquee> createState() => _PlusCoversMarqueeState();
}

class _PlusCoversMarqueeState extends State<_PlusCoversMarquee>
    with SingleTickerProviderStateMixin {
  static const _coverW = 72.0;
  static const _coverH = 108.0;
  static const _gap = 8.0;

  late AnimationController _controller;
  late double _loopWidth;

  @override
  void initState() {
    super.initState();
    _loopWidth = kBooks.length * (_coverW + _gap);
    _controller = AnimationController(
      vsync: this,
      duration: Duration(milliseconds: (_loopWidth * 28).round()),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  List<Book> get _strip => [...kBooks, ...kBooks];

  Widget _row(List<Book> books, {double topPad = 0, double leftPad = 0}) {
    return Padding(
      padding: EdgeInsets.only(top: topPad, left: leftPad),
      child: Row(
        children: books
            .map(
              (b) => Padding(
                padding: const EdgeInsets.only(right: _gap),
                child: BookCover(book: b, width: _coverW, height: _coverH),
              ),
            )
            .toList(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final shifted = [
      ...kBooks.skip(kBooks.length ~/ 2),
      ...kBooks,
    ];
    final shiftedDup = [...shifted, ...shifted];

    return SizedBox(
      height: 230,
      width: double.infinity,
      child: ClipRect(
        child: AnimatedBuilder(
          animation: _controller,
          builder: (_, __) {
            final dx = -_controller.value * _loopWidth;
            return Transform.translate(
              offset: Offset(dx, 0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _row(_strip),
                  const SizedBox(height: 10),
                  _row(shiftedDup, leftPad: 36),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════
// BOOK DETAIL
// ══════════════════════════════════════════════════════════
class BookDetailScreen extends StatefulWidget {
  final Book book;

  const BookDetailScreen({super.key, required this.book});

  @override
  State<BookDetailScreen> createState() => _BookDetailScreenState();
}

class _BookDetailScreenState extends State<BookDetailScreen> {
  bool exp = false;

  @override
  Widget build(BuildContext context) {
    final b = widget.book;
    return Scaffold(
      backgroundColor: C.bg,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 320,
            pinned: true,
            backgroundColor: b.coverColor,
            leading: Padding(
              padding: const EdgeInsets.only(left: 8),
              child: CircleAvatar(
                backgroundColor: Colors.black26,
                child: IconButton(
                  icon: const Icon(
                    Icons.arrow_back_ios_rounded,
                    color: Colors.white,
                    size: 18,
                  ),
                  onPressed: () => Navigator.pop(context),
                ),
              ),
            ),
            actions: [
              CircleAvatar(
                backgroundColor: Colors.black26,
                child: IconButton(
                  icon: Icon(
                    b.isBookmarked ? Icons.bookmark : Icons.bookmark_border,
                    color: Colors.white,
                    size: 20,
                  ),
                  onPressed: () {
                    setState(() => b.isBookmarked = !b.isBookmarked);
                    FS.saveBookmark(b.id, b.isBookmarked);
                  },
                ),
              ),
              const SizedBox(width: 4),
              CircleAvatar(
                backgroundColor: Colors.black26,
                child: IconButton(
                  icon: const Icon(
                    Icons.share_outlined,
                    color: Colors.white,
                    size: 20,
                  ),
                  onPressed: () {},
                ),
              ),
              const SizedBox(width: 8),
            ],
            flexibleSpace: FlexibleSpaceBar(
              background: Container(
                color: b.coverColor,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const SizedBox(height: 60),
                    BookCover(book: b, width: 120, height: 170),
                    const SizedBox(height: 16),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 24),
                      child: Text(
                        b.title,
                        textAlign: TextAlign.center,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      b.author,
                      style: const TextStyle(
                          color: Colors.white70, fontSize: 13),
                    ),
                  ],
                ),
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      _DC(
                        icon: Icons.star_rounded,
                        color: const Color(0xFFFFB300),
                        text: '${b.rating}',
                      ),
                      _DC(
                        icon: Icons.reviews_outlined,
                        color: C.txtGray,
                        text: fmt(b.reviews),
                      ),
                      _DC(
                        icon: Icons.menu_book_rounded,
                        color: C.txtGray,
                        text: '${b.pages} стр',
                      ),
                      if (b.isAudio)
                        _DC(
                          icon: Icons.headphones_rounded,
                          color: const Color(0xFF9C27B0),
                          text: 'Аудио',
                        ),
                      _DC(icon: null, color: C.txtGray, text: b.genre),
                    ],
                  ),
                  const SizedBox(height: 16),
                  GestureDetector(
                    onTap: () => setState(() => exp = !exp),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          b.description,
                          maxLines: exp ? null : 3,
                          overflow: exp ? null : TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontSize: 14,
                            color: C.txt,
                            height: 1.7,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          exp ? 'Скрыть' : 'Читать далее',
                          style: const TextStyle(
                            fontSize: 13,
                            color: Colors.blue,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                  if (b.readPercent > 0 && b.readPercent < 100) ...[
                    const SizedBox(height: 20),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          '${b.readPercent}% прочитано',
                          style: const TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        Text(
                          '${100 - b.readPercent}% осталось',
                          style: const TextStyle(
                              color: C.txtGray, fontSize: 12),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: LinearProgressIndicator(
                        value: b.readPercent / 100,
                        backgroundColor: C.bgGray,
                        valueColor: const AlwaysStoppedAnimation(C.txt),
                        minHeight: 6,
                      ),
                    ),
                  ],
                  if (b.readPercent == 100) ...[
                    const SizedBox(height: 16),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: const Color(0xFFEAF7EE),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Row(
                        children: [
                          Icon(Icons.check_circle_rounded,
                              color: C.green, size: 20),
                          SizedBox(width: 8),
                          Text(
                            'Книга прочитана',
                            style: TextStyle(
                              color: C.green,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                  const SizedBox(height: 24),
                  if (b.readPercent > 0 && b.readPercent < 100)
                    _CTA(
                      label: 'Продолжить чтение',
                      onTap: () => Navigator.push(
                          context, slide(ReaderScreen(book: b))),
                    )
                  else if (b.readPercent == 100)
                    _CTA(
                      label: 'Читать снова',
                      outline: true,
                      onTap: () => Navigator.push(
                          context, slide(ReaderScreen(book: b))),
                    )
                  else if (b.isFree)
                    _CTA(
                      label: 'Читать бесплатно',
                      onTap: () => Navigator.push(
                          context, slide(ReaderScreen(book: b))),
                    )
                  else
                    Column(
                      children: [
                        _CTA(
                          label: 'Читать за ${b.price.toInt()} ₸',
                          onTap: () => Navigator.push(
                              context, slide(ReaderScreen(book: b))),
                        ),
                        if (b.isAudio) ...[
                          const SizedBox(height: 10),
                          _CTA(
                            label: '🎧 Слушать аудиокнигу',
                            outline: true,
                            onTap: () {},
                          ),
                        ],
                      ],
                    ),
                  const SizedBox(height: 32),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _DC extends StatelessWidget {
  final IconData? icon;
  final Color color;
  final String text;

  const _DC({this.icon, required this.color, required this.text});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: C.bgGray,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (icon != null) ...[
              Icon(icon, size: 13, color: color),
              const SizedBox(width: 4),
            ],
            Text(
              text,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
          ],
        ),
      );
}

class _CTA extends StatelessWidget {
  final String label;
  final bool outline;
  final VoidCallback onTap;

  const _CTA(
      {required this.label, this.outline = false, required this.onTap});

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: onTap,
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 15),
          decoration: BoxDecoration(
            color: outline ? Colors.transparent : C.txt,
            borderRadius: BorderRadius.circular(28),
            border: outline ? Border.all(color: C.txt, width: 1.5) : null,
          ),
          child: Text(
            label,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w700,
              color: outline ? C.txt : Colors.white,
            ),
          ),
        ),
      );
}

// ══════════════════════════════════════════════════════════
// READER
// ══════════════════════════════════════════════════════════
class ReaderScreen extends StatefulWidget {
  final Book book;

  const ReaderScreen({super.key, required this.book});

  @override
  State<ReaderScreen> createState() => _ReaderScreenState();
}

class _ReaderScreenState extends State<ReaderScreen> {
  bool showUI = true;
  double fontSize = 16;
  bool dark = false;
  late int pct;

  static const _txt = '''
Однажды весной, в час небывало жаркого заката, в Москве, на Патриарших прудах, появились два гражданина. Первый из них — приблизительно сорокалетний, одетый в серенькую летнюю пару, — был маленького роста, упитан, лыс, свою приличную шляпу пирожком нёс в руке, а на хорошо выбритом лице его помещались сверхъестественных размеров очки в чёрной роговой оправе.

Второй — плечистый, рыжеватый, вихрастый молодой человек в заломленной на затылок клетчатой кепке — был в ковбойке, жёваных белых брюках и в чёрных тапочках.

Первый был не кто иной, как Михаил Александрович Берлиоз, редактор толстого художественного журнала и председатель правления одной крупной московской литературной ассоциации, сокращённо именуемой МАССОЛИТ, а молодой спутник его — поэт Иван Николаевич Понырев, пишущий под псевдонимом Бездомный.

— Дайте нарзану, — попросил Берлиоз.
— Нарзану нету, — ответила женщина в будочке и почему-то обиделась.
— Пиво есть? — сиплым голосом осведомился Бездомный.
— Пиво привезут к вечеру, — ответила женщина.
— А что есть? — спросил Берлиоз.
— Абрикосовая, только тёплая, — сказала женщина.
— Ну, давайте, давайте, давайте!..
''';

  @override
  void initState() {
    super.initState();
    pct = widget.book.readPercent > 0 ? widget.book.readPercent : 1;
  }

  @override
  Widget build(BuildContext context) {
    final bg = dark ? const Color(0xFF1A1A1A) : Colors.white;
    final tc = dark ? const Color(0xFFDDDDDD) : const Color(0xFF1A1A1A);
    final b = widget.book;

    return Scaffold(
      backgroundColor: bg,
      body: GestureDetector(
        onTap: () => setState(() => showUI = !showUI),
        child: Stack(
          children: [
            NotificationListener<ScrollNotification>(
              onNotification: (n) {
                if (n is ScrollUpdateNotification &&
                    n.metrics.maxScrollExtent > 0) {
                  final np =
                      ((n.metrics.pixels / n.metrics.maxScrollExtent) *
                                  98 +
                              1)
                          .round()
                          .clamp(1, 99);
                  if (np != pct) {
                    setState(() => pct = np);
                    FS.saveProgress(b.id, np);
                    FS.updateStats(np ~/ 10 + 1);
                  }
                }
                return false;
              },
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(24, 80, 24, 100),
                child: Text(
                  _txt * 8,
                  style:
                      TextStyle(fontSize: fontSize, color: tc, height: 1.75),
                ),
              ),
            ),

            // top bar
            AnimatedOpacity(
              opacity: showUI ? 1.0 : 0.0,
              duration: const Duration(milliseconds: 200),
              child: IgnorePointer(
                ignoring: !showUI,
                child: Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [bg, bg.withValues(alpha: 0)],
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                    ),
                  ),
                  child: SafeArea(
                    child: Row(
                      children: [
                        IconButton(
                          icon: Icon(
                            Icons.arrow_back_ios_rounded,
                            color: tc,
                            size: 20,
                          ),
                          onPressed: () => Navigator.pop(context),
                        ),
                        Expanded(
                          child: Text(
                            b.title,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w600,
                              color: tc,
                            ),
                          ),
                        ),
                        IconButton(
                          icon: Icon(
                            Icons.brightness_6_outlined,
                            color: tc,
                            size: 20,
                          ),
                          onPressed: () => setState(() => dark = !dark),
                        ),
                        IconButton(
                          icon: Icon(
                            Icons.text_fields_rounded,
                            color: tc,
                            size: 20,
                          ),
                          onPressed: () => _settings(context),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),

            // bottom progress
            Positioned(
              bottom: 0,
              left: 0,
              right: 0,
              child: AnimatedOpacity(
                opacity: showUI ? 1.0 : 0.0,
                duration: const Duration(milliseconds: 200),
                child: IgnorePointer(
                  ignoring: !showUI,
                  child: Container(
                    padding: EdgeInsets.fromLTRB(
                      20,
                      12,
                      20,
                      MediaQuery.of(context).padding.bottom + 12,
                    ),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [bg.withValues(alpha: 0), bg],
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                      ),
                    ),
                    child: Row(
                      children: [
                        Text(
                          '$pct%',
                          style: TextStyle(
                            color: tc.withValues(alpha: 0.5),
                            fontSize: 11,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(4),
                            child: LinearProgressIndicator(
                              value: pct / 100,
                              backgroundColor: tc.withValues(alpha: 0.1),
                              valueColor: AlwaysStoppedAnimation(tc),
                              minHeight: 3,
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Text(
                          'стр ${b.pages ~/ 2}',
                          style: TextStyle(
                            color: tc.withValues(alpha: 0.5),
                            fontSize: 11,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _settings(BuildContext ctx) => showModalBottomSheet(
        context: ctx,
        backgroundColor: Colors.white,
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
        ),
        builder: (_) => StatefulBuilder(
          builder: (_, ss) => Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'Настройки чтения',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 20),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text('Размер текста',
                        style: TextStyle(fontSize: 14)),
                    Row(
                      children: [
                        IconButton(
                          icon: const Icon(Icons.remove_circle_outline,
                              color: C.txt),
                          onPressed: () {
                            if (fontSize > 12) {
                              setState(() => fontSize--);
                              ss(() {});
                            }
                          },
                        ),
                        Text(
                          '${fontSize.toInt()}',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.add_circle_outline,
                              color: C.txt),
                          onPressed: () {
                            if (fontSize < 24) {
                              setState(() => fontSize++);
                              ss(() {});
                            }
                          },
                        ),
                      ],
                    ),
                  ],
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text('Тёмная тема',
                        style: TextStyle(fontSize: 14)),
                    Switch(
                      value: dark,
                      activeThumbColor: C.txt,
                      onChanged: (v) {
                        setState(() => dark = v);
                        ss(() {});
                      },
                    ),
                  ],
                ),
                const SizedBox(height: 8),
              ],
            ),
          ),
        ),
      );
}

// ══════════════════════════════════════════════════════════
// SHARED WIDGETS
// ══════════════════════════════════════════════════════════
class BookCover extends StatelessWidget {
  final Book book;
  final double width, height;
  final bool square;

  const BookCover({
    super.key,
    required this.book,
    required this.width,
    required this.height,
    this.square = false,
  });

  @override
  Widget build(BuildContext context) {
    final ts = square ? width * 0.09 : width * 0.1;
    return Container(
      width: width,
      height: height,
      decoration: BoxDecoration(
        color: book.coverColor,
        borderRadius: BorderRadius.circular(10),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.12),
            blurRadius: 8,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        fit: StackFit.expand,
        children: [
          if (!square)
            Positioned(
              top: height * 0.08,
              left: 0,
              right: 0,
              child: Text(
                book.author.toUpperCase(),
                textAlign: TextAlign.center,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.75),
                  fontSize: width * 0.07,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.5,
                ),
              ),
            ),
          Center(
            child: Padding(
              padding: EdgeInsets.symmetric(horizontal: width * 0.08),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (square)
                    Text(
                      book.coverEmoji,
                      style: TextStyle(fontSize: width * 0.28),
                    ),
                  if (!square) SizedBox(height: height * 0.06),
                  Text(
                    book.title,
                    textAlign: TextAlign.center,
                    maxLines: square ? 3 : 4,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: ts,
                      fontWeight: FontWeight.w800,
                      height: 1.15,
                      letterSpacing: -0.2,
                    ),
                  ),
                ],
              ),
            ),
          ),
          if (book.isFree)
            Positioned(
              top: 8,
              left: 8,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.65),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: const Text(
                  '0+',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 9,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class BookRow extends StatelessWidget {
  final Book book;
  final VoidCallback onTap;

  const BookRow({super.key, required this.book, required this.onTap});

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: onTap,
        child: Container(
          margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: C.border),
          ),
          child: Row(
            children: [
              BookCover(book: book, width: 50, height: 70),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      book.title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: C.txt,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      book.author,
                      style: const TextStyle(
                          fontSize: 12, color: C.txtGray),
                    ),
                    const SizedBox(height: 4),
                    Row(
                      children: [
                        const Icon(Icons.star_rounded,
                            size: 13, color: Color(0xFFFFB300)),
                        const SizedBox(width: 3),
                        Text(
                          '${book.rating}',
                          style: const TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        const SizedBox(width: 6),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 6, vertical: 2),
                          decoration: BoxDecoration(
                            color: C.bgGray,
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: Text(
                            book.genre,
                            style: const TextStyle(
                                fontSize: 10, color: C.txtGray),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  if (book.isFree)
                    const Text(
                      'Бесплатно',
                      style: TextStyle(
                        color: C.green,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                      ),
                    )
                  else
                    Text(
                      '${book.price.toInt()} ₸',
                      style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  if (book.isAudio) ...[
                    const SizedBox(height: 4),
                    const Icon(Icons.headphones_rounded,
                        size: 14, color: C.txtGray),
                  ],
                ],
              ),
            ],
          ),
        ),
      );
}

class _ChipsRow extends StatelessWidget {
  final List<String> labels;

  const _ChipsRow({required this.labels});

  @override
  Widget build(BuildContext context) => SizedBox(
        height: 44,
        child: ListView(
          scrollDirection: Axis.horizontal,
          padding: const EdgeInsets.fromLTRB(16, 4, 16, 8),
          children: [
            Container(
              width: 40,
              height: 36,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: C.border),
              ),
              child: const Icon(Icons.grid_view_rounded,
                  size: 18, color: C.txt),
            ),
            const SizedBox(width: 8),
            ...labels.map(
              (l) => Padding(
                padding: const EdgeInsets.only(right: 8),
                child: Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: C.border),
                  ),
                  child: Text(
                    l,
                    style: const TextStyle(
                        fontSize: 14, color: C.txt, height: 1.1),
                  ),
                ),
              ),
            ),
          ],
        ),
      );
}

class _HeroCarousel extends StatelessWidget {
  final List<Book> books;
  final String addLabel;
  final bool showToggle;

  const _HeroCarousel({
    required this.books,
    required this.addLabel,
    this.showToggle = false,
  });

  @override
  Widget build(BuildContext context) {
    if (books.isEmpty) return const SizedBox.shrink();
    return Column(
      children: [
        Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [Colors.white, C.pinkGlow, Colors.white],
              stops: [0.0, 0.55, 1.0],
            ),
          ),
          child: SizedBox(
            height: 268,
            child: PageView.builder(
              controller: PageController(viewportFraction: 0.72),
              itemCount: books.length,
              itemBuilder: (ctx, i) {
                final b = books[i];
                return Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 6),
                  child: GestureDetector(
                    onTap: () => Navigator.push(
                        ctx, slide(BookDetailScreen(book: b))),
                    child: Stack(
                      alignment: Alignment.center,
                      children: [
                        BookCover(book: b, width: 168, height: 248),
                        if (showToggle && i == 0)
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 10, vertical: 4),
                            decoration: BoxDecoration(
                              color:
                                  Colors.white.withValues(alpha: 0.92),
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Container(
                                  width: 36,
                                  height: 20,
                                  decoration: BoxDecoration(
                                    color: C.bgGray,
                                    borderRadius:
                                        BorderRadius.circular(10),
                                  ),
                                  alignment: Alignment.centerLeft,
                                  padding: const EdgeInsets.all(2),
                                  child: Container(
                                    width: 16,
                                    height: 16,
                                    decoration: const BoxDecoration(
                                      color: Colors.white,
                                      shape: BoxShape.circle,
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 6),
                                const Text(
                                  'OFF',
                                  style: TextStyle(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w700,
                                    color: C.txtGray,
                                  ),
                                ),
                              ],
                            ),
                          ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 4),
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(vertical: 15),
            decoration: BoxDecoration(
              color: C.btnDark,
              borderRadius: BorderRadius.circular(28),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.add, color: Colors.white, size: 18),
                const SizedBox(width: 6),
                Text(
                  addLabel,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _CoversRow extends StatelessWidget {
  final List<Book> books;

  const _CoversRow({required this.books});

  @override
  Widget build(BuildContext context) => ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        itemCount: books.length,
        separatorBuilder: (_, __) => const SizedBox(width: 10),
        itemBuilder: (ctx, i) {
          final b = books[i];
          return GestureDetector(
            onTap: () =>
                Navigator.push(ctx, slide(BookDetailScreen(book: b))),
            child: BookCover(book: b, width: 108, height: 160),
          );
        },
      );
}

class _AudioRow extends StatelessWidget {
  final List<Book> books;

  const _AudioRow({required this.books});

  @override
  Widget build(BuildContext context) => ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        itemCount: books.length,
        separatorBuilder: (_, __) => const SizedBox(width: 10),
        itemBuilder: (ctx, i) {
          final b = books[i];
          return GestureDetector(
            onTap: () =>
                Navigator.push(ctx, slide(BookDetailScreen(book: b))),
            child: BookCover(book: b, width: 148, height: 148, square: true),
          );
        },
      );
}

class _SuperhitsRow extends StatelessWidget {
  final List<Book> books;

  const _SuperhitsRow({required this.books});

  @override
  Widget build(BuildContext context) => ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        itemCount: books.length,
        separatorBuilder: (_, __) => const SizedBox(width: 12),
        itemBuilder: (ctx, i) {
          final b = books[i];
          final w = MediaQuery.of(ctx).size.width * 0.44;
          return GestureDetector(
            onTap: () =>
                Navigator.push(ctx, slide(BookDetailScreen(book: b))),
            child: BookCover(book: b, width: w, height: 220, square: true),
          );
        },
      );
}

class _SH extends StatelessWidget {
  final String title;
  final String? action;
  final VoidCallback? onAction;

  const _SH({required this.title, this.action, this.onAction});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.baseline,
          textBaseline: TextBaseline.alphabetic,
          children: [
            Expanded(
              child: Text(
                title,
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                  color: C.txt,
                  letterSpacing: -0.3,
                ),
              ),
            ),
            if (action != null)
              GestureDetector(
                onTap: onAction,
                child: Text(
                  action!,
                  style: const TextStyle(fontSize: 15, color: C.txtGray),
                ),
              ),
          ],
        ),
      );
}

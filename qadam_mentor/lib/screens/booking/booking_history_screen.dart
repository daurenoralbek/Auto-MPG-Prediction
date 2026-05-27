import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../config/app_colors.dart';
import '../../providers/auth_provider.dart';
import '../../providers/booking_provider.dart';
import '../../widgets/booking_card.dart';
import '../../widgets/common/empty_state_widget.dart';

class BookingHistoryScreen extends StatefulWidget {
  const BookingHistoryScreen({super.key});

  @override
  State<BookingHistoryScreen> createState() => _BookingHistoryScreenState();
}

class _BookingHistoryScreenState extends State<BookingHistoryScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabCtrl;

  @override
  void initState() {
    super.initState();
    _tabCtrl = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final bookings = context.watch<BookingProvider>();

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text('Брондау тарихы'),
        bottom: TabBar(
          controller: _tabCtrl,
          labelColor: AppColors.primary,
          unselectedLabelColor: AppColors.textHint,
          indicatorColor: AppColors.primary,
          labelStyle: GoogleFonts.nunito(
              fontWeight: FontWeight.w700, fontSize: 13),
          tabs: const [
            Tab(text: 'Күтуде'),
            Tab(text: 'Белсенді'),
            Tab(text: 'Аяқталған'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabCtrl,
        children: [
          _BookingList(
            bookings: bookings.pendingBookings,
            onDelete: (id) async {
              final ok = await context.read<BookingProvider>().deleteBooking(id);
              if (context.mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(ok ? 'Брондау жойылды' : 'Жою мүмкін болмады'),
                    backgroundColor: ok ? AppColors.success : AppColors.error,
                  ),
                );
              }
            },
            emptyTitle: 'Күтудегі брондау жоқ',
            emptySubtitle: 'Менторға сұрау жіберіңіз',
          ),
          _BookingList(
            bookings: bookings.acceptedBookings,
            emptyTitle: 'Белсенді брондау жоқ',
            emptySubtitle: 'Мақұлданған брондаулар осында көрсетіледі',
          ),
          _BookingList(
            bookings: bookings.completedBookings,
            emptyTitle: 'Аяқталған брондау жоқ',
            emptySubtitle: 'Өткен кеңестеріңіз осында сақталады',
          ),
        ],
      ),
    );
  }
}

class _BookingList extends StatelessWidget {
  final List bookings;
  final void Function(String id)? onDelete;
  final String emptyTitle;
  final String emptySubtitle;

  const _BookingList({
    required this.bookings,
    this.onDelete,
    required this.emptyTitle,
    required this.emptySubtitle,
  });

  @override
  Widget build(BuildContext context) {
    if (bookings.isEmpty) {
      return EmptyStateWidget(
        icon: Icons.event_note_outlined,
        title: emptyTitle,
        subtitle: emptySubtitle,
      );
    }
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: bookings.length,
      itemBuilder: (ctx, i) => BookingCard(
        booking: bookings[i],
        onDelete: onDelete != null ? () => onDelete!(bookings[i].id) : null,
      ),
    );
  }
}

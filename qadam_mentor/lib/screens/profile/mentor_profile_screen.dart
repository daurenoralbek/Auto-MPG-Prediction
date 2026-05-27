import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../config/app_colors.dart';
import '../../config/app_constants.dart';
import '../../config/app_routes.dart';
import '../../providers/auth_provider.dart';
import '../../providers/booking_provider.dart';
import '../../services/firestore_service.dart';
import '../../widgets/booking_card.dart';
import '../../widgets/common/empty_state_widget.dart';

// Profile screen shown to a mentor when they are logged in with role=mentor.
// Accessed from StudentProfileScreen when role == 'mentor'.
class MentorProfileScreen extends StatefulWidget {
  const MentorProfileScreen({super.key});

  @override
  State<MentorProfileScreen> createState() => _MentorProfileScreenState();
}

class _MentorProfileScreenState extends State<MentorProfileScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabCtrl;
  final _firestoreService = FirestoreService();
  bool _isAvailable = true;

  @override
  void initState() {
    super.initState();
    _tabCtrl = TabController(length: 2, vsync: this);
    _loadAvailability();
  }

  Future<void> _loadAvailability() async {
    final uid = context.read<AuthProvider>().user?.uid;
    if (uid == null) return;
    final mentor = await _firestoreService.getMentor(uid);
    if (mounted && mentor != null) {
      setState(() => _isAvailable = mentor.isAvailable);
    }
  }

  Future<void> _toggleAvailability(bool value) async {
    final uid = context.read<AuthProvider>().user?.uid;
    if (uid == null) return;
    setState(() => _isAvailable = value);
    await _firestoreService.updateMentor(uid, {'isAvailable': value});
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            value
                ? 'Бос деп белгіленді'
                : 'Бос емес деп белгіленді',
          ),
          backgroundColor:
              value ? AppColors.success : AppColors.pending,
        ),
      );
    }
  }

  @override
  void dispose() {
    _tabCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final user = context.watch<AuthProvider>().user;
    final bookings = context.watch<BookingProvider>().bookings;

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text('Ментор панелі'),
        actions: [
          IconButton(
            icon: const Icon(Icons.edit_outlined),
            onPressed: () =>
                Navigator.pushNamed(context, AppRoutes.editProfile),
          ),
        ],
        bottom: TabBar(
          controller: _tabCtrl,
          labelColor: AppColors.primary,
          unselectedLabelColor: AppColors.textHint,
          indicatorColor: AppColors.primary,
          tabs: const [
            Tab(text: 'Профиль'),
            Tab(text: 'Брондаулар'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabCtrl,
        children: [
          // Profile tab
          SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Center(
                  child: Column(
                    children: [
                      Container(
                        width: 80,
                        height: 80,
                        decoration: BoxDecoration(
                          gradient: AppColors.primaryGradient,
                          shape: BoxShape.circle,
                          border: Border.all(
                              color: AppColors.white, width: 3),
                        ),
                        child: Center(
                          child: Text(
                            user?.name.isNotEmpty == true
                                ? user!.name[0].toUpperCase()
                                : '?',
                            style: GoogleFonts.nunito(
                              fontSize: 32,
                              fontWeight: FontWeight.w800,
                              color: AppColors.white,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        user?.name ?? '',
                        style: GoogleFonts.nunito(
                          fontSize: 20,
                          fontWeight: FontWeight.w800,
                          color: AppColors.textPrimary,
                        ),
                      ),
                      Text(
                        user?.specialty ?? 'Ментор',
                        style: GoogleFonts.nunito(
                          fontSize: 13,
                          color: AppColors.primary,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                // Availability toggle
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 16, vertical: 14),
                  decoration: BoxDecoration(
                    color: AppColors.white,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: AppColors.divider),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 40,
                        height: 40,
                        decoration: BoxDecoration(
                          color: _isAvailable
                              ? AppColors.successLight
                              : AppColors.errorLight,
                          shape: BoxShape.circle,
                        ),
                        child: Icon(
                          Icons.circle,
                          size: 16,
                          color: _isAvailable
                              ? AppColors.success
                              : AppColors.error,
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Қол жетімділік',
                              style: GoogleFonts.nunito(
                                fontSize: 14,
                                fontWeight: FontWeight.w700,
                                color: AppColors.textPrimary,
                              ),
                            ),
                            Text(
                              _isAvailable
                                  ? 'Студенттер брондай алады'
                                  : 'Брондау уақытша жабық',
                              style: GoogleFonts.nunito(
                                fontSize: 12,
                                color: AppColors.textSecondary,
                              ),
                            ),
                          ],
                        ),
                      ),
                      Switch(
                        value: _isAvailable,
                        onChanged: _toggleAvailability,
                        activeColor: AppColors.success,
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),

                // Stats
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: AppColors.white,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: AppColors.divider),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _StatItem(
                          value: bookings.length.toString(),
                          label: 'Барлық сұрау'),
                      Container(width: 1, height: 36, color: AppColors.divider),
                      _StatItem(
                          value: bookings
                              .where((b) => b.status == 'pending')
                              .length
                              .toString(),
                          label: 'Күтуде'),
                      Container(width: 1, height: 36, color: AppColors.divider),
                      _StatItem(
                          value: bookings
                              .where((b) => b.status == 'completed')
                              .length
                              .toString(),
                          label: 'Аяқталды'),
                    ],
                  ),
                ),
                const SizedBox(height: 12),

                // Mentor categories
                if (user?.specialty != null) ...[
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: AppColors.white,
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(color: AppColors.divider),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Мамандық',
                          style: GoogleFonts.nunito(
                            fontSize: 12,
                            color: AppColors.textSecondary,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          user!.specialty!,
                          style: GoogleFonts.nunito(
                            fontSize: 15,
                            fontWeight: FontWeight.w700,
                            color: AppColors.textPrimary,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ],
            ),
          ),

          // Bookings tab (mentor sees requests)
          bookings.isEmpty
              ? const EmptyStateWidget(
                  icon: Icons.event_note_outlined,
                  title: 'Брондау сұраулары жоқ',
                  subtitle: 'Студенттердің сұраулары осында көрсетіледі',
                )
              : ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: bookings.length,
                  itemBuilder: (ctx, i) => BookingCard(
                    booking: bookings[i],
                    showMentorInfo: false,
                  ),
                ),
        ],
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  final String value;
  final String label;
  const _StatItem({required this.value, required this.label});

  @override
  Widget build(BuildContext context) => Column(
        children: [
          Text(
            value,
            style: GoogleFonts.nunito(
                fontSize: 20,
                fontWeight: FontWeight.w800,
                color: AppColors.textPrimary),
          ),
          Text(
            label,
            style: GoogleFonts.nunito(
                fontSize: 11, color: AppColors.textSecondary),
          ),
        ],
      );
}

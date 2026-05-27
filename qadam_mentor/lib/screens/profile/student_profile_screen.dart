import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../config/app_colors.dart';
import '../../config/app_routes.dart';
import '../../providers/auth_provider.dart';
import '../../providers/booking_provider.dart';
import '../../providers/favorites_provider.dart';

class StudentProfileScreen extends StatelessWidget {
  const StudentProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final user = context.watch<AuthProvider>().user;
    final bookings = context.watch<BookingProvider>().bookings;
    final favCount = context.watch<FavoritesProvider>().favoriteIds.length;

    return Scaffold(
      backgroundColor: AppColors.background,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 220,
            pinned: true,
            backgroundColor: AppColors.primary,
            actions: [
              IconButton(
                icon: const Icon(Icons.edit_outlined, color: AppColors.white),
                onPressed: () => Navigator.pushNamed(context, AppRoutes.editProfile),
              ),
            ],
            flexibleSpace: FlexibleSpaceBar(
              background: Container(
                decoration: const BoxDecoration(gradient: AppColors.primaryGradient),
                child: SafeArea(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const SizedBox(height: 20),
                      _ProfileAvatar(
                        photoUrl: user?.photoUrl,
                        name: user?.name ?? '',
                      ),
                      const SizedBox(height: 10),
                      Text(
                        user?.name ?? '',
                        style: GoogleFonts.nunito(
                          fontSize: 20,
                          fontWeight: FontWeight.w800,
                          color: AppColors.white,
                        ),
                      ),
                      Text(
                        user?.email ?? '',
                        style: GoogleFonts.nunito(
                          fontSize: 13,
                          color: AppColors.white.withOpacity(0.8),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),

          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  // Stats
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: AppColors.white,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: AppColors.divider),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _Stat(value: bookings.length.toString(), label: 'Брондау'),
                        Container(width: 1, height: 40, color: AppColors.divider),
                        _Stat(
                          value: bookings.where((b) => b.status == 'completed').length.toString(),
                          label: 'Аяқталды',
                        ),
                        Container(width: 1, height: 40, color: AppColors.divider),
                        _Stat(value: favCount.toString(), label: 'Таңдаулы'),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Info card
                  _InfoCard(
                    children: [
                      if (user?.specialty != null && user!.specialty!.isNotEmpty)
                        _InfoRow(Icons.work_outline_rounded, 'Мамандық', user.specialty!),
                      if (user?.year != null && user!.year!.isNotEmpty)
                        _InfoRow(Icons.school_outlined, 'Курс', '${user.year} курс'),
                      if (user?.phone != null && user!.phone!.isNotEmpty)
                        _InfoRow(Icons.phone_outlined, 'Телефон', user.phone!),
                    ],
                  ),
                  const SizedBox(height: 8),

                  // Menu
                  _MenuCard(children: [
                    _MenuItem(
                      icon: Icons.history_rounded,
                      label: 'Брондау тарихы',
                      onTap: () => Navigator.pushNamed(context, AppRoutes.bookingHistory),
                    ),
                    _MenuItem(
                      icon: Icons.favorite_border_rounded,
                      label: 'Таңдаулы менторлар',
                      onTap: () => Navigator.pushNamed(context, AppRoutes.favorites),
                    ),
                    _MenuItem(
                      icon: Icons.notifications_outlined,
                      label: 'Хабарландырулар',
                      onTap: () => Navigator.pushNamed(context, AppRoutes.notifications),
                    ),
                    _MenuItem(
                      icon: Icons.settings_outlined,
                      label: 'Параметрлер',
                      onTap: () => Navigator.pushNamed(context, AppRoutes.settings),
                    ),
                    _MenuItem(
                      icon: Icons.help_outline_rounded,
                      label: 'Анықтама',
                      onTap: () => Navigator.pushNamed(context, AppRoutes.helpCenter),
                    ),
                  ]),
                  const SizedBox(height: 8),

                  _MenuCard(children: [
                    _MenuItem(
                      icon: Icons.logout_rounded,
                      label: 'Шығу',
                      color: AppColors.error,
                      onTap: () async {
                        final confirmed = await showDialog<bool>(
                          context: context,
                          builder: (_) => AlertDialog(
                            title: Text('Шығу',
                                style: GoogleFonts.nunito(fontWeight: FontWeight.w700)),
                            content: Text('Жүйеден шығуды растайсыз ба?',
                                style: GoogleFonts.nunito()),
                            actions: [
                              TextButton(
                                  onPressed: () => Navigator.pop(context, false),
                                  child: const Text('Болдырмау')),
                              TextButton(
                                  onPressed: () => Navigator.pop(context, true),
                                  child: Text('Шығу',
                                      style: GoogleFonts.nunito(color: AppColors.error))),
                            ],
                          ),
                        );
                        if (confirmed == true && context.mounted) {
                          await context.read<AuthProvider>().signOut();
                          if (context.mounted) {
                            Navigator.pushReplacementNamed(context, AppRoutes.login);
                          }
                        }
                      },
                    ),
                  ]),
                  const SizedBox(height: 24),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ProfileAvatar extends StatelessWidget {
  final String? photoUrl;
  final String name;
  const _ProfileAvatar({this.photoUrl, required this.name});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 82,
      height: 82,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        border: Border.all(color: AppColors.white, width: 3),
      ),
      child: ClipOval(
        child: photoUrl != null && photoUrl!.isNotEmpty
            ? CachedNetworkImage(
                imageUrl: photoUrl!,
                fit: BoxFit.cover,
                errorWidget: (_, __, ___) => _fallback(),
              )
            : _fallback(),
      ),
    );
  }

  Widget _fallback() => Container(
        color: AppColors.white.withOpacity(0.3),
        child: Center(
          child: Text(
            name.isNotEmpty ? name[0].toUpperCase() : '?',
            style: GoogleFonts.nunito(
                fontSize: 32, fontWeight: FontWeight.w800, color: AppColors.white),
          ),
        ),
      );
}

class _Stat extends StatelessWidget {
  final String value;
  final String label;
  const _Stat({required this.value, required this.label});

  @override
  Widget build(BuildContext context) => Column(
        children: [
          Text(value, style: GoogleFonts.nunito(fontSize: 22, fontWeight: FontWeight.w800, color: AppColors.textPrimary)),
          Text(label, style: GoogleFonts.nunito(fontSize: 12, color: AppColors.textSecondary)),
        ],
      );
}

class _InfoCard extends StatelessWidget {
  final List<Widget> children;
  const _InfoCard({required this.children});

  @override
  Widget build(BuildContext context) {
    if (children.isEmpty) return const SizedBox.shrink();
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: AppColors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.divider),
      ),
      child: Column(children: children),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  const _InfoRow(this.icon, this.label, this.value);

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Row(
          children: [
            Icon(icon, size: 18, color: AppColors.primary),
            const SizedBox(width: 12),
            Text(label, style: GoogleFonts.nunito(fontSize: 13, color: AppColors.textSecondary)),
            const Spacer(),
            Text(value, style: GoogleFonts.nunito(fontSize: 13, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
          ],
        ),
      );
}

class _MenuCard extends StatelessWidget {
  final List<Widget> children;
  const _MenuCard({required this.children});

  @override
  Widget build(BuildContext context) => Container(
        decoration: BoxDecoration(
          color: AppColors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.divider),
        ),
        child: Column(children: children),
      );
}

class _MenuItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final Color? color;
  const _MenuItem({required this.icon, required this.label, required this.onTap, this.color});

  @override
  Widget build(BuildContext context) => ListTile(
        leading: Icon(icon, color: color ?? AppColors.textSecondary, size: 22),
        title: Text(label, style: GoogleFonts.nunito(fontSize: 14, fontWeight: FontWeight.w600, color: color ?? AppColors.textPrimary)),
        trailing: color == null ? const Icon(Icons.chevron_right_rounded, color: AppColors.textHint) : null,
        onTap: onTap,
        dense: true,
      );
}

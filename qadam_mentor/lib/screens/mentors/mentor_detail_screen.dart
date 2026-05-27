import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../config/app_colors.dart';
import '../../config/app_constants.dart';
import '../../config/app_routes.dart';
import '../../models/mentor_model.dart';
import '../../providers/auth_provider.dart';
import '../../providers/chat_provider.dart';
import '../../providers/favorites_provider.dart';

class MentorDetailScreen extends StatelessWidget {
  const MentorDetailScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final mentor = ModalRoute.of(context)!.settings.arguments as MentorModel;
    final uid = context.read<AuthProvider>().user?.uid ?? '';
    final userName = context.read<AuthProvider>().user?.name ?? '';
    final userPhoto = context.read<AuthProvider>().user?.photoUrl;
    final favs = context.watch<FavoritesProvider>();
    final isFav = favs.isFavorite(mentor.id);

    return Scaffold(
      backgroundColor: AppColors.background,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 280,
            pinned: true,
            backgroundColor: AppColors.primary,
            leading: IconButton(
              icon: const Icon(Icons.arrow_back_ios_rounded,
                  color: AppColors.white),
              onPressed: () => Navigator.pop(context),
            ),
            actions: [
              IconButton(
                icon: Icon(
                  isFav ? Icons.favorite_rounded : Icons.favorite_border_rounded,
                  color: isFav ? Colors.red[300] : AppColors.white,
                ),
                onPressed: () =>
                    context.read<FavoritesProvider>().toggle(uid, mentor.id),
              ),
            ],
            flexibleSpace: FlexibleSpaceBar(
              background: Stack(
                fit: StackFit.expand,
                children: [
                  mentor.photoUrl != null && mentor.photoUrl!.isNotEmpty
                      ? CachedNetworkImage(
                          imageUrl: mentor.photoUrl!,
                          fit: BoxFit.cover,
                          placeholder: (_, __) =>
                              Container(color: AppColors.primaryLight),
                          errorWidget: (_, __, ___) =>
                              _AvatarBg(name: mentor.name),
                        )
                      : _AvatarBg(name: mentor.name),
                  Container(
                    decoration: const BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [Colors.transparent, Colors.black54],
                      ),
                    ),
                  ),
                  Positioned(
                    bottom: 16,
                    left: 20,
                    right: 20,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          mentor.name,
                          style: GoogleFonts.nunito(
                            fontSize: 22,
                            fontWeight: FontWeight.w800,
                            color: AppColors.white,
                          ),
                        ),
                        Text(
                          '${mentor.roleLabel} · ${mentor.specialty}',
                          style: GoogleFonts.nunito(
                            fontSize: 14,
                            color: AppColors.white.withOpacity(0.85),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),

          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Stats Row
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 14),
                    decoration: BoxDecoration(
                      color: AppColors.white,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: AppColors.divider),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _StatItem(
                          value: mentor.rating.toStringAsFixed(1),
                          label: 'Рейтинг',
                          icon: Icons.star_rounded,
                          iconColor: const Color(0xFFFFC107),
                        ),
                        _Divider(),
                        _StatItem(
                          value: mentor.reviewCount.toString(),
                          label: 'Пікір',
                          icon: Icons.rate_review_outlined,
                          iconColor: AppColors.primary,
                        ),
                        _Divider(),
                        _StatItem(
                          value: mentor.isAvailable ? 'Бос' : 'Бос емес',
                          label: 'Күй',
                          icon: Icons.circle,
                          iconColor: mentor.isAvailable
                              ? AppColors.success
                              : AppColors.error,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Bio
                  if (mentor.bio.isNotEmpty) ...[
                    _SectionTitle('Ментор туралы'),
                    const SizedBox(height: 8),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(14),
                      decoration: BoxDecoration(
                        color: AppColors.white,
                        borderRadius: BorderRadius.circular(14),
                        border: Border.all(color: AppColors.divider),
                      ),
                      child: Text(
                        mentor.bio,
                        style: GoogleFonts.nunito(
                          fontSize: 14,
                          color: AppColors.textSecondary,
                          height: 1.6,
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                  ],

                  // Categories
                  if (mentor.categories.isNotEmpty) ...[
                    _SectionTitle('Бағыттар'),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: mentor.categories.map((c) {
                        final cat = AppConstants.mentorCategories
                            .firstWhere((m) => m['id'] == c,
                                orElse: () => {'label': c, 'icon': '📌'});
                        return Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 12, vertical: 6),
                          decoration: BoxDecoration(
                            color: AppColors.primarySurface,
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            '${cat['icon']} ${cat['label']}',
                            style: GoogleFonts.nunito(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: AppColors.primary,
                            ),
                          ),
                        );
                      }).toList(),
                    ),
                    const SizedBox(height: 16),
                  ],

                  // Schedule
                  _SectionTitle('Кесте'),
                  const SizedBox(height: 8),
                  Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: AppColors.white,
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(color: AppColors.divider),
                    ),
                    child: Column(
                      children: [
                        _ScheduleRow(
                          icon: Icons.access_time_rounded,
                          label: 'Уақыт',
                          value: mentor.availableHours,
                        ),
                        const SizedBox(height: 8),
                        _ScheduleRow(
                          icon: Icons.calendar_month_outlined,
                          label: 'Күндер',
                          value: mentor.availableDays.join(', '),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),

                  // Action Buttons
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () async {
                            final chat = context.read<ChatProvider>();
                            final chatId = await chat.getOrCreateChat(
                              studentId: uid,
                              mentorId: mentor.uid,
                              studentName: userName,
                              mentorName: mentor.name,
                              mentorPhoto: mentor.photoUrl,
                              studentPhoto: userPhoto,
                            );
                            if (context.mounted) {
                              Navigator.pushNamed(
                                context,
                                AppRoutes.chatDetail,
                                arguments: {
                                  'chatId': chatId,
                                  'mentor': mentor,
                                },
                              );
                            }
                          },
                          icon: const Icon(Icons.chat_bubble_outline_rounded),
                          label: const Text('Хабар жазу'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: mentor.isAvailable
                              ? () => Navigator.pushNamed(
                                    context,
                                    AppRoutes.booking,
                                    arguments: mentor,
                                  )
                              : null,
                          icon: const Icon(Icons.event_available_rounded),
                          label: const Text('Брондау'),
                        ),
                      ),
                    ],
                  ),
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

class _AvatarBg extends StatelessWidget {
  final String name;
  const _AvatarBg({required this.name});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(gradient: AppColors.primaryGradient),
      child: Center(
        child: Text(
          name.isNotEmpty ? name[0].toUpperCase() : '?',
          style: GoogleFonts.nunito(
            fontSize: 80,
            fontWeight: FontWeight.w800,
            color: AppColors.white.withOpacity(0.4),
          ),
        ),
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  final String text;
  const _SectionTitle(this.text);

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: GoogleFonts.nunito(
        fontSize: 16,
        fontWeight: FontWeight.w700,
        color: AppColors.textPrimary,
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  final String value;
  final String label;
  final IconData icon;
  final Color iconColor;

  const _StatItem({
    required this.value,
    required this.label,
    required this.icon,
    required this.iconColor,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Icon(icon, color: iconColor, size: 22),
        const SizedBox(height: 4),
        Text(
          value,
          style: GoogleFonts.nunito(
            fontSize: 14,
            fontWeight: FontWeight.w800,
            color: AppColors.textPrimary,
          ),
        ),
        Text(
          label,
          style: GoogleFonts.nunito(
            fontSize: 11,
            color: AppColors.textSecondary,
          ),
        ),
      ],
    );
  }
}

class _Divider extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(width: 1, height: 40, color: AppColors.divider);
  }
}

class _ScheduleRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;

  const _ScheduleRow({
    required this.icon,
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 18, color: AppColors.primary),
        const SizedBox(width: 10),
        Text(
          '$label: ',
          style: GoogleFonts.nunito(
            fontSize: 13,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        Expanded(
          child: Text(
            value,
            style: GoogleFonts.nunito(
              fontSize: 13,
              color: AppColors.textSecondary,
            ),
          ),
        ),
      ],
    );
  }
}

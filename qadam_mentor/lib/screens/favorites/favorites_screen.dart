import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../config/app_colors.dart';
import '../../config/app_routes.dart';
import '../../providers/auth_provider.dart';
import '../../providers/favorites_provider.dart';
import '../../providers/mentor_provider.dart';
import '../../widgets/common/empty_state_widget.dart';
import '../../widgets/mentor_card.dart';

class FavoritesScreen extends StatelessWidget {
  const FavoritesScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final uid = context.read<AuthProvider>().user?.uid ?? '';
    final favIds = context.watch<FavoritesProvider>().favoriteIds;
    final allMentors = context.watch<MentorProvider>().mentors;
    final favMentors =
        allMentors.where((m) => favIds.contains(m.id)).toList();

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(title: const Text('Таңдаулылар')),
      body: favMentors.isEmpty
          ? const EmptyStateWidget(
              icon: Icons.favorite_border_rounded,
              title: 'Таңдаулы ментор жоқ',
              subtitle:
                  'Ментор профиліндегі жүрек белгісін басып таңдаулыларға қосыңыз',
            )
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: favMentors.length,
              itemBuilder: (ctx, i) {
                final m = favMentors[i];
                return MentorCard(
                  mentor: m,
                  isFavorite: true,
                  onFavoriteTap: () =>
                      context.read<FavoritesProvider>().toggle(uid, m.id),
                  onTap: () => Navigator.pushNamed(
                    ctx,
                    AppRoutes.mentorDetail,
                    arguments: m,
                  ),
                );
              },
            ),
    );
  }
}

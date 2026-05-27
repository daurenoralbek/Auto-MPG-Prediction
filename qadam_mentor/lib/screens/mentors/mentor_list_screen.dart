import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../config/app_colors.dart';
import '../../config/app_constants.dart';
import '../../config/app_routes.dart';
import '../../providers/auth_provider.dart';
import '../../providers/favorites_provider.dart';
import '../../providers/mentor_provider.dart';
import '../../widgets/common/empty_state_widget.dart';
import '../../widgets/common/loading_widget.dart';
import '../../widgets/mentor_card.dart';

class MentorListScreen extends StatefulWidget {
  const MentorListScreen({super.key});

  @override
  State<MentorListScreen> createState() => _MentorListScreenState();
}

class _MentorListScreenState extends State<MentorListScreen> {
  final _searchCtrl = TextEditingController();

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final mentors = context.watch<MentorProvider>();
    final favs = context.watch<FavoritesProvider>();
    final uid = context.read<AuthProvider>().user?.uid ?? '';

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text('Менторлар'),
        actions: [
          IconButton(
            icon: const Icon(Icons.filter_list_rounded),
            onPressed: () =>
                Navigator.pushNamed(context, AppRoutes.filter),
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 0),
            child: TextField(
              controller: _searchCtrl,
              onChanged: (v) => context.read<MentorProvider>().search(v),
              decoration: InputDecoration(
                hintText: 'Ментор іздеу...',
                prefixIcon: const Icon(Icons.search_rounded,
                    color: AppColors.textHint, size: 20),
                suffixIcon: _searchCtrl.text.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.close, size: 18),
                        onPressed: () {
                          _searchCtrl.clear();
                          context.read<MentorProvider>().search('');
                        },
                      )
                    : null,
              ),
            ),
          ),
          const SizedBox(height: 10),
          SizedBox(
            height: 38,
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              scrollDirection: Axis.horizontal,
              itemCount: AppConstants.mentorCategories.length + 1,
              itemBuilder: (ctx, i) {
                if (i == 0) {
                  final allSelected = mentors.selectedCategory == null;
                  return _CategoryChip(
                    label: 'Барлығы',
                    selected: allSelected,
                    onTap: () =>
                        context.read<MentorProvider>().filterByCategory(null),
                  );
                }
                final cat = AppConstants.mentorCategories[i - 1];
                final selected = mentors.selectedCategory == cat['id'];
                return _CategoryChip(
                  label: cat['label']!,
                  selected: selected,
                  onTap: () => context
                      .read<MentorProvider>()
                      .filterByCategory(selected ? null : cat['id']),
                );
              },
            ),
          ),
          const SizedBox(height: 8),
          Expanded(
            child: mentors.isLoading
                ? const Padding(
                    padding: EdgeInsets.all(16),
                    child: Column(children: [
                      MentorCardSkeleton(),
                      MentorCardSkeleton(),
                      MentorCardSkeleton(),
                    ]),
                  )
                : mentors.mentors.isEmpty
                    ? const EmptyStateWidget(
                        icon: Icons.people_outline_rounded,
                        title: 'Ментор табылмады',
                        subtitle: 'Іздеу сұрауын немесе сүзгіні өзгертіп көріңіз',
                      )
                    : ListView.builder(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        itemCount: mentors.mentors.length,
                        itemBuilder: (ctx, i) {
                          final m = mentors.mentors[i];
                          return MentorCard(
                            mentor: m,
                            isFavorite: favs.isFavorite(m.id),
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
          ),
        ],
      ),
    );
  }
}

class _CategoryChip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _CategoryChip({
    required this.label,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        margin: const EdgeInsets.only(right: 8),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: selected ? AppColors.primary : AppColors.white,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: selected ? AppColors.primary : AppColors.divider,
          ),
        ),
        child: Text(
          label,
          style: GoogleFonts.nunito(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: selected ? AppColors.white : AppColors.textSecondary,
          ),
        ),
      ),
    );
  }
}

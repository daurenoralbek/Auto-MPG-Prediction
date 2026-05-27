import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../config/app_colors.dart';
import '../../config/app_constants.dart';
import '../../providers/mentor_provider.dart';

class FilterScreen extends StatefulWidget {
  const FilterScreen({super.key});

  @override
  State<FilterScreen> createState() => _FilterScreenState();
}

class _FilterScreenState extends State<FilterScreen> {
  String? _selectedCategory;
  bool _availableOnly = false;
  String _selectedRole = 'all';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text('Сүзгі'),
        actions: [
          TextButton(
            onPressed: () {
              setState(() {
                _selectedCategory = null;
                _availableOnly = false;
                _selectedRole = 'all';
              });
            },
            child: Text(
              'Тазалау',
              style: GoogleFonts.nunito(
                color: AppColors.primary,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _SectionTitle('Санат'),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: AppConstants.mentorCategories.map((cat) {
              final selected = _selectedCategory == cat['id'];
              return GestureDetector(
                onTap: () => setState(() =>
                    _selectedCategory = selected ? null : cat['id']),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color:
                        selected ? AppColors.primary : AppColors.white,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: selected ? AppColors.primary : AppColors.divider,
                    ),
                  ),
                  child: Text(
                    '${cat['icon']} ${cat['label']}',
                    style: GoogleFonts.nunito(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: selected
                          ? AppColors.white
                          : AppColors.textSecondary,
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 20),
          _SectionTitle('Ментор рөлі'),
          const SizedBox(height: 10),
          ...[
            {'id': 'all', 'label': 'Барлығы'},
            {'id': 'teacher', 'label': 'Мұғалімдер'},
            {'id': 'senior_student', 'label': 'Ағалық студенттер'},
            {'id': 'club_leader', 'label': 'Клуб жетекшілері'},
          ].map((r) => RadioListTile<String>(
                value: r['id']!,
                groupValue: _selectedRole,
                onChanged: (v) => setState(() => _selectedRole = v!),
                title: Text(
                  r['label']!,
                  style: GoogleFonts.nunito(
                    fontSize: 14,
                    color: AppColors.textPrimary,
                  ),
                ),
                activeColor: AppColors.primary,
                contentPadding: EdgeInsets.zero,
              )),
          const SizedBox(height: 12),
          SwitchListTile(
            value: _availableOnly,
            onChanged: (v) => setState(() => _availableOnly = v),
            title: Text(
              'Тек бос менторлар',
              style: GoogleFonts.nunito(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: AppColors.textPrimary,
              ),
            ),
            activeColor: AppColors.primary,
            contentPadding: EdgeInsets.zero,
          ),
          const SizedBox(height: 28),
          ElevatedButton(
            onPressed: () {
              context
                  .read<MentorProvider>()
                  .filterByCategory(_selectedCategory);
              Navigator.pop(context);
            },
            child: const Text('Қолдану'),
          ),
        ],
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

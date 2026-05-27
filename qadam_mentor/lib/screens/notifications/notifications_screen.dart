import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../config/app_colors.dart';
import '../../widgets/common/empty_state_widget.dart';

class NotificationsScreen extends StatelessWidget {
  const NotificationsScreen({super.key});

  // Sample static notifications (in a real app these come from Firestore)
  static const _items = [
    _NotifData(
      icon: Icons.check_circle_outline_rounded,
      color: AppColors.success,
      bg: AppColors.successLight,
      title: 'Сұрауыңыз қабылданды',
      subtitle: 'Айжан Нұрмаханова сіздің брондауыңызды мақұлдады.',
      time: '2 сағат бұрын',
    ),
    _NotifData(
      icon: Icons.chat_bubble_outline_rounded,
      color: AppColors.primary,
      bg: AppColors.primarySurface,
      title: 'Жаңа хабар',
      subtitle: 'Ментордан жаңа хабар келді.',
      time: '5 сағат бұрын',
    ),
    _NotifData(
      icon: Icons.event_available_rounded,
      color: AppColors.secondary,
      bg: Color(0xFFE1F5FE),
      title: 'Кеңес ертең',
      subtitle: 'Ертеңгі кеңесіңізді ұмытпаңыз: сағат 14:00.',
      time: '1 күн бұрын',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(title: const Text('Хабарландырулар')),
      body: _items.isEmpty
          ? const EmptyStateWidget(
              icon: Icons.notifications_none_rounded,
              title: 'Хабарландыру жоқ',
              subtitle: 'Жаңа хабарландырулар осында көрсетіледі',
            )
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _items.length,
              itemBuilder: (_, i) => _NotifCard(data: _items[i]),
            ),
    );
  }
}

class _NotifData {
  final IconData icon;
  final Color color;
  final Color bg;
  final String title;
  final String subtitle;
  final String time;

  const _NotifData({
    required this.icon,
    required this.color,
    required this.bg,
    required this.title,
    required this.subtitle,
    required this.time,
  });
}

class _NotifCard extends StatelessWidget {
  final _NotifData data;
  const _NotifCard({required this.data});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.divider),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: data.bg,
              shape: BoxShape.circle,
            ),
            child: Icon(data.icon, color: data.color, size: 22),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  data.title,
                  style: GoogleFonts.nunito(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  data.subtitle,
                  style: GoogleFonts.nunito(
                    fontSize: 13,
                    color: AppColors.textSecondary,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  data.time,
                  style: GoogleFonts.nunito(
                    fontSize: 11,
                    color: AppColors.textHint,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

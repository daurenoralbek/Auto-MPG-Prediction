import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:timeago/timeago.dart' as timeago;
import '../../config/app_colors.dart';
import '../../config/app_routes.dart';
import '../../providers/auth_provider.dart';
import '../../providers/chat_provider.dart';
import '../../widgets/common/empty_state_widget.dart';

class ChatListScreen extends StatelessWidget {
  const ChatListScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final uid = context.read<AuthProvider>().user?.uid ?? '';
    final chats = context.watch<ChatProvider>().chats;

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(title: const Text('Хабарлар')),
      body: chats.isEmpty
          ? const EmptyStateWidget(
              icon: Icons.chat_bubble_outline_rounded,
              title: 'Хабарлар жоқ',
              subtitle: 'Ментормен сөйлесу үшін профиліне кіріңіз',
            )
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: chats.length,
              itemBuilder: (ctx, i) {
                final chat = chats[i];
                final name = chat.otherPersonName(uid);
                final photo = chat.otherPersonPhoto(uid);

                return GestureDetector(
                  onTap: () => Navigator.pushNamed(
                    ctx,
                    AppRoutes.chatDetail,
                    arguments: {
                      'chatId': chat.id,
                      'otherName': name,
                      'otherPhoto': photo,
                    },
                  ),
                  child: Container(
                    margin: const EdgeInsets.only(bottom: 10),
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: AppColors.white,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: AppColors.divider),
                    ),
                    child: Row(
                      children: [
                        _Avatar(photo: photo, name: name),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                name,
                                style: GoogleFonts.nunito(
                                  fontSize: 15,
                                  fontWeight: FontWeight.w700,
                                  color: AppColors.textPrimary,
                                ),
                              ),
                              const SizedBox(height: 3),
                              Text(
                                chat.lastMessage.isEmpty
                                    ? 'Сөйлесу басталды'
                                    : chat.lastMessage,
                                style: GoogleFonts.nunito(
                                  fontSize: 13,
                                  color: AppColors.textSecondary,
                                ),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ],
                          ),
                        ),
                        Text(
                          timeago.format(chat.lastMessageTime, locale: 'kk'),
                          style: GoogleFonts.nunito(
                            fontSize: 11,
                            color: AppColors.textHint,
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
    );
  }
}

class _Avatar extends StatelessWidget {
  final String? photo;
  final String name;
  const _Avatar({this.photo, required this.name});

  @override
  Widget build(BuildContext context) {
    if (photo != null && photo!.isNotEmpty) {
      return ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: CachedNetworkImage(
          imageUrl: photo!,
          width: 52,
          height: 52,
          fit: BoxFit.cover,
          errorWidget: (_, __, ___) => _fallback(),
        ),
      );
    }
    return _fallback();
  }

  Widget _fallback() {
    return Container(
      width: 52,
      height: 52,
      decoration: BoxDecoration(
        gradient: AppColors.primaryGradient,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Center(
        child: Text(
          name.isNotEmpty ? name[0].toUpperCase() : '?',
          style: GoogleFonts.nunito(
            fontSize: 20,
            fontWeight: FontWeight.w800,
            color: AppColors.white,
          ),
        ),
      ),
    );
  }
}
